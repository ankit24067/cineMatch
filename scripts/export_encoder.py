"""Export MiniLM's weights and vocabulary for the dependency-free runtime encoder.

`Recommender.search` needs a 384-d vector for a free-text query, which means
running the sentence-transformers model at request time. Doing that with torch
would put a ~900 MB dependency into the serving image, and doing it with ONNX
costs 124 MB of runtime plus a 90 MB fp32 graph (§14). Neither is proportionate
to encoding one short string per request.

So this script dumps the raw tensors instead. `api/encoder.py` reimplements the
forward pass in numpy, which the serving image already depends on. Weights are
stored as float16: the encoder upcasts to float32 before any arithmetic, so the
only cost is a rounding of the stored constants, and §9 records the measured
parity against sentence-transformers.

The pooler is deliberately not exported. sentence-transformers mean-pools the
final hidden states and never reads `pooler.dense`, so shipping it would be
590 KB of weight that nothing multiplies.

Run after any change to the embedding model, and rebuild `content_emb.npy` in
the same pass -- a query encoder and a document matrix from different models
produce silently meaningless cosines.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "encoder"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256


def _state_dict(model_name: str) -> tuple[dict[str, np.ndarray], dict]:
    """The BertModel tensors behind the sentence-transformers wrapper.

    Imported lazily. torch and sentence-transformers are development-only
    dependencies and must never be reachable from the serving image.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    inner = None
    for module in model[0].modules():
        if type(module).__name__ == "BertModel":
            inner = module
            break
    if inner is None:
        raise RuntimeError(f"no BertModel inside {model_name}")

    pooling = model[1].get_config_dict()
    mode = pooling.get("pooling_mode")
    mean_tokens = pooling.get("pooling_mode_mean_tokens")
    if (mode is not None and mode != "mean") or mean_tokens is False:
        raise RuntimeError(f"encoder assumes mean pooling, model reports {pooling}")

    tensors = {k: v.detach().numpy() for k, v in inner.state_dict().items()}
    return tensors, inner.config.to_dict()


def _vocabulary(model_name: str) -> list[str]:
    """Vocabulary ordered by token id, and a check that it is the one we assume."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backend = json.loads(tokenizer.backend_tokenizer.to_str())

    normalizer = backend["normalizer"]
    if not normalizer.get("lowercase"):
        raise RuntimeError("encoder assumes a lowercasing normaliser")
    if backend["model"]["type"] != "WordPiece":
        raise RuntimeError(f"encoder assumes WordPiece, got {backend['model']['type']}")
    if backend["model"].get("continuing_subword_prefix") != "##":
        raise RuntimeError("encoder assumes the ## continuation prefix")

    vocab = tokenizer.get_vocab()
    ordered = [""] * len(vocab)
    for token, index in vocab.items():
        ordered[index] = token
    return ordered


def export(model_name: str = MODEL_NAME, out_dir: Path = ARTIFACT_DIR) -> dict:
    """Write weights.npz, vocab.txt and config.json. Returns the build report."""
    tensors, config = _state_dict(model_name)
    vocab = _vocabulary(model_name)

    weights = {
        name: tensor.astype(np.float16)
        for name, tensor in tensors.items()
        if not name.startswith("pooler.")
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "weights.npz", **weights)
    (out_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")

    settings = {
        "model": model_name,
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "intermediate_size": config["intermediate_size"],
        "layer_norm_eps": config["layer_norm_eps"],
        "max_position_embeddings": config["max_position_embeddings"],
        "max_seq_length": MAX_SEQ_LENGTH,
        "vocab_size": len(vocab),
        "pad_token_id": config["pad_token_id"],
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "pooling": "mean",
        "normalise": True,
    }
    (out_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    n_params = sum(int(np.prod(w.shape)) for w in weights.values())
    return {
        "tensors": len(weights),
        "parameters": n_params,
        "weights_mb": round((out_dir / "weights.npz").stat().st_size / 1e6, 1),
        "vocab_size": len(vocab),
    }


def main() -> None:
    report = export()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
