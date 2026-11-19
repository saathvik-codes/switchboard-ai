"""
Sentence embeddings via a real HuggingFace transformer (mean-pooled token
embeddings from `sentence-transformers/all-MiniLM-L6-v2`), loaded directly
through `transformers.AutoModel`/`AutoTokenizer` rather than the
`sentence-transformers` package's own wrapper - same model weights, same
embeddings, just a dependency-lighter import path.
"""
import os

# transformers lazily imports TensorFlow as an optional backend even though
# we only need PyTorch - on environments with a broken/mismatched TF or
# protobuf install (common when other ML packages have pulled in
# conflicting versions), that import alone crashes model loading. We don't
# use TF anywhere, so tell transformers not to bother.
os.environ.setdefault("USE_TF", "0")

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        # imported here, not at the top: the API and agent import this module, and torch/transformers (a ~2 GB
        # install) are only needed when a message is actually embedded, so tests and tooling can run without them
        from transformers import AutoModel, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _mean_pool(token_embeddings, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    import torch
    tokenizer, model = _load()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.numpy())
    return np.vstack(all_embeddings)


if __name__ == "__main__":
    vecs = embed(["I lost my card", "What's my account balance?", "I lost my card somewhere"])
    print("shape:", vecs.shape)
    print("cosine(lost card, balance):", float(vecs[0] @ vecs[1]))
    print("cosine(lost card, lost card variant):", float(vecs[0] @ vecs[2]))
