"""A dependency-free MiniLM text encoder: WordPiece and a BERT forward pass in numpy.

The recommender scores a free-text query by cosine against `content_emb.npy`,
which means the query has to be embedded by the same model that built that
matrix. The serving image cannot carry torch, and the ONNX runtime that was the
original plan costs 124 MB of dependency for what amounts to six small matrix
multiplies per request (section 14).

So the forward pass lives here instead, over the float16 tensors written by
`scripts/export_encoder.py`. numpy and scipy are already serving dependencies,
so semantic search adds no new ones. Parity against sentence-transformers is
measured in `tests/test_encoder.py` and quoted in section 9.

This is an inference-only reimplementation: no dropout, no attention output, no
pooler. It reproduces `BertModel` for the settings all-MiniLM-L6-v2 actually
uses, namely absolute position embeddings, a single token type and exact-erf
GELU, and `export_encoder.py` fails loudly if the checkpoint stops matching
those assumptions.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import erf

ENCODER_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "encoder"


def _is_control(char: str) -> bool:
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char).startswith("C")


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    if (33 <= code <= 47) or (58 <= code <= 64) or (91 <= code <= 96) or (123 <= code <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


class WordPieceTokenizer:
    """BertTokenizer's lowercase, accent-stripping WordPiece, in pure Python.

    Reproduces the pipeline the exported checkpoint was trained with: clean the
    text, isolate CJK characters, lowercase and strip accents, split off
    punctuation, then greedily match the longest vocabulary prefix at each
    position, falling back to [UNK] for any word that cannot be covered.
    """

    def __init__(self, vocab: dict[str, int], max_input_chars_per_word: int = 100) -> None:
        self.vocab = vocab
        self.unk_id = vocab["[UNK]"]
        self.cls_id = vocab["[CLS]"]
        self.sep_id = vocab["[SEP]"]
        self.pad_id = vocab["[PAD]"]
        self.max_input_chars_per_word = max_input_chars_per_word

    @staticmethod
    def _clean(text: str) -> str:
        out = []
        for char in text:
            if ord(char) in (0, 0xFFFD) or _is_control(char):
                continue
            out.append(" " if char.isspace() else char)
        return "".join(out)

    @staticmethod
    def _space_chinese(text: str) -> str:
        out = []
        for char in text:
            if _is_chinese(char):
                out.extend((" ", char, " "))
            else:
                out.append(char)
        return "".join(out)

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")

    @staticmethod
    def _split_punctuation(word: str) -> list[str]:
        pieces: list[str] = []
        current: list[str] = []
        for char in word:
            if _is_punctuation(char):
                if current:
                    pieces.append("".join(current))
                    current = []
                pieces.append(char)
            else:
                current.append(char)
        if current:
            pieces.append("".join(current))
        return pieces

    def basic_tokenize(self, text: str) -> list[str]:
        cleaned = self._space_chinese(self._clean(text))
        tokens: list[str] = []
        for word in cleaned.split():
            lowered = self._strip_accents(word.lower())
            tokens.extend(self._split_punctuation(lowered))
        return tokens

    def wordpiece(self, token: str) -> list[int]:
        if len(token) > self.max_input_chars_per_word:
            return [self.unk_id]
        pieces: list[int] = []
        start = 0
        while start < len(token):
            end = len(token)
            found = None
            while start < end:
                candidate = token[start:end]
                if start > 0:
                    candidate = "##" + candidate
                if candidate in self.vocab:
                    found = self.vocab[candidate]
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            pieces.append(found)
            start = end
        return pieces

    def encode(self, text: str, max_length: int) -> list[int]:
        """Token ids with [CLS] and [SEP], truncated to `max_length` including both."""
        ids: list[int] = []
        for token in self.basic_tokenize(text):
            ids.extend(self.wordpiece(token))
        ids = ids[: max_length - 2]
        return [self.cls_id, *ids, self.sep_id]


@dataclass(frozen=True)
class _Layer:
    """One transformer block, with every projection pre-transposed for x @ W.

    Every array here is float32 and stays float32. A single float64 scalar
    leaking into the attention arithmetic promotes the whole activation chain,
    and each subsequent matmul then upcasts a fresh copy of the weight it is
    multiplying against -- which cost 180 ms per query before `scale` was
    pre-cast.
    """

    q_w: np.ndarray
    q_b: np.ndarray
    k_w: np.ndarray
    k_b: np.ndarray
    v_w: np.ndarray
    v_b: np.ndarray
    attn_w: np.ndarray
    attn_b: np.ndarray
    attn_ln_w: np.ndarray
    attn_ln_b: np.ndarray
    inter_w: np.ndarray
    inter_b: np.ndarray
    out_w: np.ndarray
    out_b: np.ndarray
    out_ln_w: np.ndarray
    out_ln_b: np.ndarray


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = np.mean(centred * centred, axis=-1, keepdims=True)
    return centred / np.sqrt(variance + eps) * weight + bias


_INV_SQRT2 = np.float32(1.0 / np.sqrt(2.0))
_HALF = np.float32(0.5)
_ONE = np.float32(1.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU. The checkpoint specifies hidden_act gelu, not the tanh approximation.

    The constants are pre-cast for the reason given on `_Layer`: `np.sqrt(2.0)`
    is a float64 scalar, and dividing a float32 activation by it promotes every
    matmul downstream.
    """
    return x * _HALF * (_ONE + erf(x * _INV_SQRT2))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class TextEncoder:
    """MiniLM inference over the exported tensors.

    Word embeddings stay in float16 and are upcast as they are gathered, which
    keeps 23 MB of the checkpoint out of float32 while the layer projections,
    which every token multiplies against, are converted once at load.
    """

    def __init__(self, encoder_dir: Path = ENCODER_DIR) -> None:
        config = json.loads((encoder_dir / "config.json").read_text(encoding="utf-8"))
        vocab_tokens = (encoder_dir / "vocab.txt").read_text(encoding="utf-8").split("\n")
        self.tokenizer = WordPieceTokenizer({t: i for i, t in enumerate(vocab_tokens)})

        self.dim = int(config["hidden_size"])
        self.heads = int(config["num_attention_heads"])
        self.head_dim = self.dim // self.heads
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))
        self.eps = float(config["layer_norm_eps"])
        self.max_seq_length = int(config["max_seq_length"])
        self.model_name = str(config["model"])

        with np.load(encoder_dir / "weights.npz") as data:
            self.word_embeddings = data["embeddings.word_embeddings.weight"]
            self.position_embeddings = data["embeddings.position_embeddings.weight"].astype(
                np.float32
            )
            self.token_type_embedding = data["embeddings.token_type_embeddings.weight"].astype(
                np.float32
            )[0]
            self.emb_ln_w = data["embeddings.LayerNorm.weight"].astype(np.float32)
            self.emb_ln_b = data["embeddings.LayerNorm.bias"].astype(np.float32)
            self.layers = tuple(
                self._read_layer(data, i) for i in range(int(config["num_hidden_layers"]))
            )

    @staticmethod
    def _read_layer(data, index: int) -> _Layer:
        def dense(path: str) -> tuple[np.ndarray, np.ndarray]:
            prefix = f"encoder.layer.{index}.{path}"
            return (
                data[f"{prefix}.weight"].astype(np.float32).T,
                data[f"{prefix}.bias"].astype(np.float32),
            )

        def norm(path: str) -> tuple[np.ndarray, np.ndarray]:
            prefix = f"encoder.layer.{index}.{path}"
            return (
                data[f"{prefix}.weight"].astype(np.float32),
                data[f"{prefix}.bias"].astype(np.float32),
            )

        q_w, q_b = dense("attention.self.query")
        k_w, k_b = dense("attention.self.key")
        v_w, v_b = dense("attention.self.value")
        attn_w, attn_b = dense("attention.output.dense")
        attn_ln_w, attn_ln_b = norm("attention.output.LayerNorm")
        inter_w, inter_b = dense("intermediate.dense")
        out_w, out_b = dense("output.dense")
        out_ln_w, out_ln_b = norm("output.LayerNorm")

        return _Layer(
            q_w=q_w,
            q_b=q_b,
            k_w=k_w,
            k_b=k_b,
            v_w=v_w,
            v_b=v_b,
            attn_w=attn_w,
            attn_b=attn_b,
            attn_ln_w=attn_ln_w,
            attn_ln_b=attn_ln_b,
            inter_w=inter_w,
            inter_b=inter_b,
            out_w=out_w,
            out_b=out_b,
            out_ln_w=out_ln_w,
            out_ln_b=out_ln_b,
        )

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        tokens = self.word_embeddings[ids].astype(np.float32)
        positions = self.position_embeddings[: ids.shape[1]]
        hidden = tokens + positions + self.token_type_embedding
        return _layer_norm(hidden, self.emb_ln_w, self.emb_ln_b, self.eps)

    def _attend(self, layer: _Layer, hidden: np.ndarray, bias: np.ndarray) -> np.ndarray:
        batch, length, _ = hidden.shape
        shape = (batch, length, self.heads, self.head_dim)

        query = (hidden @ layer.q_w + layer.q_b).reshape(shape).transpose(0, 2, 1, 3)
        key = (hidden @ layer.k_w + layer.k_b).reshape(shape).transpose(0, 2, 1, 3)
        value = (hidden @ layer.v_w + layer.v_b).reshape(shape).transpose(0, 2, 1, 3)

        scores = query @ key.transpose(0, 1, 3, 2) * self.scale + bias
        context = _softmax(scores) @ value
        merged = context.transpose(0, 2, 1, 3).reshape(batch, length, self.dim)

        projected = merged @ layer.attn_w + layer.attn_b
        return _layer_norm(hidden + projected, layer.attn_ln_w, layer.attn_ln_b, self.eps)

    def _block(self, layer: _Layer, hidden: np.ndarray, bias: np.ndarray) -> np.ndarray:
        attended = self._attend(layer, hidden, bias)
        intermediate = _gelu(attended @ layer.inter_w + layer.inter_b)
        projected = intermediate @ layer.out_w + layer.out_b
        return _layer_norm(attended + projected, layer.out_ln_w, layer.out_ln_b, self.eps)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised embeddings, one row per input string.

        Inputs that tokenise to nothing but the special tokens still produce a
        finite row, and a zero-norm result is returned as zeros rather than
        NaNs so `Recommender.search` can reject it on the norm.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        encoded = [self.tokenizer.encode(text, self.max_seq_length) for text in texts]
        length = max(len(ids) for ids in encoded)
        pad = self.tokenizer.pad_id

        ids = np.full((len(encoded), length), pad, dtype=np.int64)
        mask = np.zeros((len(encoded), length), dtype=np.float32)
        for row, sequence in enumerate(encoded):
            ids[row, : len(sequence)] = sequence
            mask[row, : len(sequence)] = 1.0

        bias = (1.0 - mask)[:, None, None, :] * np.float32(-1e9)

        hidden = self._embed(ids)
        for layer in self.layers:
            hidden = self._block(layer, hidden, bias)

        weights = mask[:, :, None]
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return np.divide(pooled, norms, out=np.zeros_like(pooled), where=norms > 0).astype(
            np.float32
        )

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def is_available(encoder_dir: Path = ENCODER_DIR) -> bool:
    """Whether the exported checkpoint is present, without loading it to find out."""
    return all(
        (encoder_dir / name).exists() for name in ("weights.npz", "vocab.txt", "config.json")
    )


@lru_cache(maxsize=1)
def load_encoder(encoder_dir: Path = ENCODER_DIR) -> TextEncoder:
    """The process-wide encoder, built on first use.

    Measured at 101 MB of resident memory and 0.5 s to build: 41 MB of npz
    inflating to 23 MB of float16 embeddings plus 43 MB of float32 layers, with
    the rest in the vocabulary dict. That is a real fraction of a 512 MB free
    tier, so it is deferred until something actually calls /search rather than
    paid by every deployment at import.
    """
    return TextEncoder(encoder_dir)
