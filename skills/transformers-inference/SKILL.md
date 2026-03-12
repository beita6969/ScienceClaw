---
name: transformers-inference
description: "HuggingFace Transformers for model inference. Use when: text classification, NER, question answering, summarization, embeddings, zero-shot classification. NOT for: training large models (use cloud), simple regex/rule-based tasks, production serving at scale (use vLLM)."
metadata: { "openclaw": { "emoji": "🤗", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-transformers", "kind": "uv", "package": "transformers" }, { "id": "uv-torch", "kind": "uv", "package": "torch" }, { "id": "uv-sentencepiece", "kind": "uv", "package": "sentencepiece" }] } }
---

# HuggingFace Transformers Inference

Text classification, NER, question answering, summarization, embeddings, and zero-shot classification using pretrained models.

## When to Use / When NOT to Use

**Use when:** text classification, named entity recognition, question answering, summarization, text generation, sentence embeddings, zero-shot classification, translation, fill-mask tasks.

**NOT for:** training large models from scratch (use cloud GPU clusters), simple regex or rule-based text processing, production serving at scale (use vLLM or TGI), tasks that don't need neural models.

## Quick Pipelines

```python
from transformers import pipeline

# Text classification (sentiment)
clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = clf("This movie was fantastic!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Named entity recognition
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
entities = ner("Hugging Face is based in New York City.")
# [{'entity_group': 'ORG', ...}, {'entity_group': 'LOC', ...}]

# Question answering (extractive)
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
answer = qa(question="What is the capital of France?",
            context="France is a country in Europe. Its capital is Paris.")
# {'answer': 'Paris', 'score': 0.99, ...}

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(long_text, max_length=130, min_length=30, do_sample=False)

# Zero-shot classification (no task-specific training needed)
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = zsc("I need to pay my electricity bill",
             candidate_labels=["finance", "health", "technology"])
# {'labels': ['finance', ...], 'scores': [0.95, ...]}
```

## Manual Model and Tokenizer Loading

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer("Hello world", return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: [batch, seq_len, hidden_dim]
# outputs.pooler_output:     [batch, hidden_dim]
```

## Sentence Embeddings (Mean Pooling)

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    return F.normalize(embeddings, p=2, dim=1)

embs = get_embeddings(["Hello world", "Hi there"])
similarity = torch.mm(embs, embs.T)  # cosine similarity matrix
```

## Batch Processing

```python
clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english",
               device=0 if torch.cuda.is_available() else -1)

texts = ["Great product!", "Terrible service.", "It was okay."]
results = clf(texts, batch_size=16)           # batch for throughput

# For large datasets, use dataset streaming
from transformers import pipeline
for result in clf(iter(large_text_list), batch_size=32):
    process(result)
```

## Best Practices

1. Use `pipeline()` for quick prototyping; load model/tokenizer manually for custom logic.
2. Set `device=0` for GPU or `device="mps"` on Apple Silicon; default is CPU.
3. Always use `torch.no_grad()` during inference to save memory.
4. Use `truncation=True` and `max_length` to handle long inputs safely.
5. For repeated inference, load the model once and reuse; avoid reloading per call.
6. Prefer distilled models (distilbert, distilgpt2) for faster inference when accuracy allows.
7. Use `model.half()` or `torch.float16` to reduce memory footprint on GPU.
8. Cache models locally with `TRANSFORMERS_CACHE` env var to avoid repeated downloads.
