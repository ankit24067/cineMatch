"""The numpy encoder, against the model it replaces.

Two of these tests need torch and sentence-transformers and are skipped when
those are absent, which is the case in the serving image by design. The pair
that matters most does not: `test_reproduces_committed_document_embeddings`
re-encodes the sample documents and compares them to the rows they produced in
`content_emb.npy`, which pins the runtime encoder to the artifacts it has to
agree with using nothing but numpy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from api.artifacts import ARTIFACT_DIR
from api.encoder import ENCODER_DIR, WordPieceTokenizer, is_available, load_encoder

pytestmark = pytest.mark.skipif(
    not is_available(), reason="artifacts/encoder is absent; run scripts/export_encoder.py"
)

QUERIES = [
    "space movies that are actually about grief",
    "a heist that goes wrong",
    "dystopian future where technology controls society",
    "Amelie",
    "naive Bayes 100% !!",
    "Crouching Tiger 卧虎藏龙 (2000)",
    "a" * 300,
]


@pytest.fixture(scope="module")
def encoder():
    return load_encoder()


@pytest.fixture(scope="module")
def documents() -> list[str]:
    return json.loads((ARTIFACT_DIR / "documents_sample.json").read_text(encoding="utf-8"))


def test_reproduces_committed_document_embeddings(encoder, documents):
    """The runtime encoder must agree with the matrix it will be compared against.

    `documents_sample.json` holds the first documents in item order, so their
    embeddings are the leading rows of `content_emb.npy`. A drift here means
    every /search cosine is being computed between two different vector spaces.
    """
    embeddings = np.load(ARTIFACT_DIR / "content_emb.npy")
    encoded = encoder.encode(documents)
    cosines = (encoded * embeddings[: len(documents)]).sum(axis=1)
    assert cosines.min() > 0.9999, f"encoder drifted from content_emb.npy: {cosines}"


def test_embeddings_are_unit_length(encoder):
    norms = np.linalg.norm(encoder.encode(QUERIES), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_batching_does_not_change_results(encoder):
    """Padding a short query alongside a long one must not alter its vector."""
    together = encoder.encode([QUERIES[0], QUERIES[6]])
    alone = encoder.encode_one(QUERIES[0])
    assert float(together[0] @ alone) > 0.9999


def test_truncates_at_max_sequence_length(encoder):
    ids = encoder.tokenizer.encode("word " * 5000, encoder.max_seq_length)
    assert len(ids) == encoder.max_seq_length
    assert ids[0] == encoder.tokenizer.cls_id
    assert ids[-1] == encoder.tokenizer.sep_id


def test_empty_query_yields_a_zero_vector_not_nan(encoder):
    vector = encoder.encode_one("   ")
    assert np.isfinite(vector).all()


def test_unknown_word_falls_back_to_unk(encoder):
    tokenizer = encoder.tokenizer
    assert tokenizer.wordpiece("qwxzjvk" * 20) == [tokenizer.unk_id]


def test_vocabulary_matches_the_exported_config(encoder):
    config = json.loads((ENCODER_DIR / "config.json").read_text(encoding="utf-8"))
    assert len(encoder.tokenizer.vocab) == config["vocab_size"]


def test_wordpiece_splits_on_the_continuation_prefix():
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "un": 4, "##aff": 5, "##able": 6}
    tokenizer = WordPieceTokenizer(vocab)
    assert tokenizer.wordpiece("unaffable") == [4, 5, 6]


def test_tokenizer_matches_huggingface(encoder):
    transformers = pytest.importorskip("transformers")
    reference = transformers.AutoTokenizer.from_pretrained(encoder.model_name)
    for text in QUERIES:
        expected = reference(text, truncation=True, max_length=encoder.max_seq_length)
        assert encoder.tokenizer.encode(text, encoder.max_seq_length) == expected["input_ids"], text


def test_embeddings_match_sentence_transformers(encoder, documents):
    """Parity with the model this reimplements, on queries and on real documents."""
    sentence_transformers = pytest.importorskip("sentence_transformers")
    reference = sentence_transformers.SentenceTransformer(encoder.model_name, device="cpu")

    texts = QUERIES + documents
    expected = reference.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    cosines = (encoder.encode(texts) * expected).sum(axis=1)
    assert cosines.min() > 0.9999, f"worst parity {cosines.min()} on {texts[int(cosines.argmin())]}"
