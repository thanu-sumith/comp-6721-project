"""Contracts for semantic features, safe caching and validation prediction export."""
import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embeddings import ChunkedEncoder, EmbeddingFeatures, cached_documents, unit_rows, validate_predictions


class FakeEncoder:
    dimension = 3

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        return (np.array([[len(t), 1, 2] for t in texts], dtype=np.float32),
                np.array([len(t) for t in texts]), np.ones(len(texts), dtype=int))


class EmbeddingTests(unittest.TestCase):
    def test_swap_equivalence_and_prompt_interactions(self):
        p, a, b = unit_rows([[1, 2, 0]]), unit_rows([[3, 0, 1]]), unit_rows([[0, 2, 1]])
        x = EmbeddingFeatures(p, a, b)
        np.testing.assert_allclose(x.design(True), EmbeddingFeatures(p, b, a).design())
        self.assertEqual(x.design().shape, (1, 21))
        self.assertFalse(np.array_equal(x.design(), EmbeddingFeatures(-p, a, b).design()))

    def test_empty_vector_is_finite(self):
        np.testing.assert_array_equal(unit_rows(np.zeros((2, 3))), np.zeros((2, 3)))

    def test_cache_reorders_deduplicates_and_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            enc = FakeEncoder()
            v, _, _, _ = cached_documents(["one", "two words", "one"], enc, directory, {"revision": "v1"})
            self.assertEqual(enc.calls, ["one", "two words"])
            np.testing.assert_array_equal(v[0], v[2])
            enc.calls.clear()
            reordered, _, _, _ = cached_documents(["two words", "one"], enc, directory, {"revision": "v1"})
            self.assertEqual(enc.calls, [])
            np.testing.assert_array_equal(reordered, v[[1, 0]])
            cached_documents(["one"], enc, directory, {"revision": "v2"})
            self.assertEqual(enc.calls, ["one"])

    def test_export_rejects_misaligned_ids_labels_and_invalid_probabilities(self):
        ids, y = np.array(["v1", "v2"]), np.array([0, 2])
        p = np.array([[.7, .2, .1], [.2, .2, .6]])
        validate_predictions(ids, p, y, ids, y)
        bad_cases = [(ids[::-1], p, y), (ids, p, y[::-1]),
                     (ids, np.ones((2, 3)), y), (ids, np.full((2, 3), np.nan), y),
                     (np.array(["v1", "v1"]), p, y), (ids, p[:, :2], y)]
        for actual_ids, probs, labels in bad_cases:
            with self.subTest(actual_ids=actual_ids, probs=probs):
                with self.assertRaises(ValueError):
                    validate_predictions(actual_ids, probs, labels, ids, y)

    def test_cache_resumes_after_interruption(self):
        class InterruptedEncoder(FakeEncoder):
            def encode(self, texts):
                if self.calls:
                    raise RuntimeError("interrupted after the first committed block")
                return super().encode(texts)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                cached_documents(["first", "second"], InterruptedEncoder(), directory, {}, block_size=1)
            resumed = FakeEncoder()
            vectors, _, _, _ = cached_documents(["first", "second"], resumed, directory, {}, block_size=1)
            self.assertEqual(resumed.calls, ["second"])
            np.testing.assert_array_equal(vectors[:, 0], [5, 6])

    def test_chunking_retains_each_content_token_exactly_once(self):
        import torch

        class Tokenizer:
            cls_token_id, sep_token_id = 1, 2

            def encode(self, text, **kwargs):
                return [int(v) for v in text.split()]

            def pad(self, inputs, **kwargs):
                sequences = inputs["input_ids"]
                width = max(map(len, sequences))
                return {"input_ids": torch.tensor([s+[0]*(width-len(s)) for s in sequences])}

        class EncoderModel:
            device = "cpu"

            def __init__(self):
                self.seen = []

            def __call__(self, batch):
                vectors = []
                for row in batch["input_ids"].numpy():
                    content = row[row > 2]
                    self.seen.extend(content.tolist())
                    vectors.append([float(content.sum()), float(len(content)), 1.0])
                return {"sentence_embedding": torch.tensor(vectors)}

        encoder = ChunkedEncoder.__new__(ChunkedEncoder)
        encoder.torch, encoder.tokenizer, encoder.model = torch, Tokenizer(), EncoderModel()
        encoder.content_tokens, encoder.batch_size, encoder.dimension = 6, 2, 3
        original = list(range(10, 30))
        vectors, tokens, chunks = encoder.encode([" ".join(map(str, original)), ""])
        self.assertEqual(encoder.model.seen, original)
        np.testing.assert_array_equal(tokens, [20, 0])
        np.testing.assert_array_equal(chunks, [4, 0])
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1.0, places=5)
        np.testing.assert_array_equal(vectors[1], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
