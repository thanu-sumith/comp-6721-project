"""Frozen MiniLM document embeddings and prompt-aware comparison features.

Long documents use non-overlapping token chunks. Normalized chunk embeddings are
averaged by content-token count and normalized again; empty text maps to zero.
Only explicitly supplied texts are encoded. The caller excludes local test rows.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
POLICY = "token_chunks_length_weighted_mean_l2_v1"


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class ChunkedEncoder:
    """Load a pinned encoder once; perform inference without updating any weights."""

    def __init__(self, revision=MODEL_REVISION, max_tokens=256, batch_size=64,
                 device=None, model_cache=None, local_files_only=False):
        import torch
        from sentence_transformers import SentenceTransformer

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.torch = torch
        self.model = SentenceTransformer(
            MODEL_NAME, revision=revision, device=device,
            cache_folder=str(model_cache) if model_cache else None,
            local_files_only=local_files_only, trust_remote_code=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = self.model.tokenizer
        if not 8 <= max_tokens <= self.model.max_seq_length:
            raise ValueError(f"max_tokens must be between 8 and {self.model.max_seq_length}")
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.content_tokens = max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.tokenizer.num_special_tokens_to_add(pair=False) != 2 or self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
            raise ValueError("This pinned MiniLM encoder requires the BERT CLS/content/SEP token layout")
        dimension_fn = getattr(self.model, "get_embedding_dimension", None)
        self.dimension = dimension_fn() if dimension_fn else self.model.get_sentence_embedding_dimension()
        self.device = str(self.model.device)

    def encode(self, texts):
        """Return document vectors, content-token counts and chunk counts."""
        chunks, owners, weights = [], [], []
        token_counts = np.zeros(len(texts), dtype=np.int64)
        chunk_counts = np.zeros(len(texts), dtype=np.int64)
        for i, text in enumerate(texts):
            # No truncation: every content token belongs to exactly one chunk.
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            token_counts[i] = len(tokens)
            for start in range(0, len(tokens), self.content_tokens):
                content = tokens[start:start + self.content_tokens]
                # The pinned BERT-style MiniLM uses exactly CLS + content + SEP.
                # Construct from IDs directly so subwords at boundaries are not re-tokenized.
                chunks.append([self.tokenizer.cls_token_id, *content, self.tokenizer.sep_token_id])
                owners.append(i)
                weights.append(len(content))
                chunk_counts[i] += 1
        pooled = np.zeros((len(texts), self.dimension), dtype=np.float32)
        # Group similar lengths to reduce padding. Token order inside each chunk stays intact.
        order = sorted(range(len(chunks)), key=lambda k: len(chunks[k]), reverse=True)
        with self.torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                pick = order[start:start + self.batch_size]
                batch = self.tokenizer.pad(
                    {"input_ids": [chunks[k] for k in pick]}, padding=True, return_tensors="pt")
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                vectors = self.model(batch)["sentence_embedding"].float().cpu().numpy()
                vectors = unit_rows(vectors)
                for k, vector in zip(pick, vectors):
                    pooled[owners[k]] += weights[k] * vector
        pooled /= np.maximum(token_counts[:, None], 1)
        return unit_rows(pooled), token_counts, chunk_counts


def cached_documents(texts, encoder, cache_dir, configuration, block_size=128):
    """Cache by exact text and encoder configuration, independently of row ordering.

    Committed blocks survive interruption. No raw text or preference labels are
    stored in the cache. Dataset IDs and labels are handled by the experiment.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"documents_{signature}.sqlite3"
    keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    unique = dict(zip(keys, texts))
    found = {}
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, v BLOB, tokens INTEGER, chunks INTEGER)")
        missing = []
        for key, text in unique.items():
            row = con.execute("SELECT v, tokens, chunks FROM vectors WHERE key=?", (key,)).fetchone()
            if row is None:
                missing.append((key, text))
            else:
                v = np.frombuffer(row[0], dtype="<f4").copy()
                if v.shape != (encoder.dimension,) or not np.isfinite(v).all():
                    raise ValueError(f"Invalid embedding cache entry in {path}")
                found[key] = (v, int(row[1]), int(row[2]))
        print(f"Embedding cache: {len(unique)-len(missing):,} hits; {len(missing):,} unique texts to encode", flush=True)
        for start in range(0, len(missing), block_size):
            block = missing[start:start + block_size]
            vectors, tokens, counts = encoder.encode([t for _, t in block])
            for (key, _), v, nt, nc in zip(block, vectors, tokens, counts):
                con.execute("INSERT INTO vectors VALUES (?, ?, ?, ?)",
                            (key, np.asarray(v, dtype="<f4").tobytes(), int(nt), int(nc)))
                found[key] = (v, int(nt), int(nc))
            con.commit()
            if start == 0 or (start // block_size) % 8 == 0 or start + block_size >= len(missing):
                print(f"Encoded {min(start+block_size, len(missing)):,}/{len(missing):,} new texts", flush=True)
    return (np.stack([found[k][0] for k in keys]),
            np.array([found[k][1] for k in keys]),
            np.array([found[k][2] for k in keys]), str(path))


@dataclass
class EmbeddingFeatures:
    prompt: np.ndarray
    a: np.ndarray
    b: np.ndarray

    def take(self, idx):
        return EmbeddingFeatures(self.prompt[idx], self.a[idx], self.b[idx])

    def design(self, swap=False):
        a, b = (self.b, self.a) if swap else (self.a, self.b)
        p = self.prompt
        similarities = np.column_stack([(p*a).sum(1), (p*b).sum(1), (a*b).sum(1)])
        # Products let a linear classifier use prompt/answer interactions.
        return np.hstack([p, a, b, np.abs(a-b), p*a, p*b, similarities]).astype(np.float32)


def validate_predictions(ids, probabilities, labels, expected_ids, expected_labels):
    """Require complete validation coverage, aligned labels and valid A/B/tie probabilities."""
    ids = np.asarray(ids).astype(str)
    expected_ids = np.asarray(expected_ids).astype(str)
    p, y = np.asarray(probabilities), np.asarray(labels)
    if len(set(ids)) != len(ids) or not np.array_equal(ids, expected_ids):
        raise ValueError("Prediction IDs must match the full validation partition in order")
    if not np.array_equal(y, expected_labels):
        raise ValueError("Prediction labels do not match the validation labels")
    if p.shape != (len(ids), 3) or not np.isfinite(p).all():
        raise ValueError("Expected finite probabilities with shape (validation rows, 3)")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError("Invalid probability values or row sums")
