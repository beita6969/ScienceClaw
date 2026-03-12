---
name: spacy-nlp
description: "Natural language processing with spaCy. Use when: (1) named entity recognition, (2) POS tagging and dependency parsing, (3) text tokenization and linguistic analysis, (4) rule-based pattern matching, (5) custom NER pipelines. NOT for: LLM inference or text generation (use transformers), sentiment analysis at scale (use dedicated models), or machine translation."
metadata: { "openclaw": { "emoji": "📝", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-spacy", "kind": "uv", "package": "spacy" }] } }
---

# spaCy NLP

Natural language processing using spaCy for tokenization, named entity
recognition, dependency parsing, and linguistic analysis.

## When to Use

- Named entity recognition (NER) on text
- Part-of-speech tagging and morphological analysis
- Dependency parsing and syntactic analysis
- Rule-based pattern matching in text
- Custom NER pipeline creation
- Tokenization and sentence segmentation
- Lemmatization and linguistic feature extraction

## When NOT to Use

- LLM inference or text generation (use transformers/huggingface)
- Sentiment analysis at scale (use fine-tuned classifiers)
- Machine translation (use dedicated MT models)
- Topic modeling (use gensim or sklearn)
- Simple regex-only text search (use re module)

## Setup and Model Download

```bash
# Download models (run once before using)
python3 -m spacy download en_core_web_sm    # small, fast, ~12MB
python3 -m spacy download en_core_web_md    # medium with word vectors, ~40MB
python3 -m spacy download en_core_web_trf   # transformer-based, most accurate
```

## Basic Pipeline

```python
import spacy

# Load a model
nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")

# Token-level attributes
for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.dep_:10} {token.lemma_}")
# Apple        PROPN  nsubj      Apple
# is           AUX    aux        be
# looking      VERB   ROOT       look
# ...
```

## Named Entity Recognition

```python
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.")

# Extract entities
for ent in doc.ents:
    print(f"{ent.text:25} {ent.label_:10} {ent.start_char}-{ent.end_char}")
# Apple Inc.                ORG        0-10
# Steve Jobs                PERSON     26-36
# Cupertino, California     GPE        40-61
# 1976                      DATE       65-69

# Common entity labels: PERSON, ORG, GPE, DATE, MONEY, PRODUCT, EVENT, LOC

# Get explanation of labels
print(spacy.explain("GPE"))   # "Countries, cities, states"
```

## Dependency Parsing

```python
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

# Dependency tree
for token in doc:
    print(f"{token.text:10} --{token.dep_:10}--> {token.head.text}")

# Noun chunks (base noun phrases)
for chunk in doc.noun_chunks:
    print(f"{chunk.text:25} root={chunk.root.text}, head={chunk.root.head.text}")
# The quick brown fox       root=fox, head=jumps
# the lazy dog              root=dog, head=over

# Find subject and object of a verb
for token in doc:
    if token.dep_ == "nsubj":
        print(f"Subject: {token.text} of verb: {token.head.text}")
    if token.dep_ == "dobj":
        print(f"Object: {token.text} of verb: {token.head.text}")
```

## Pattern Matching

```python
from spacy.matcher import Matcher, PhraseMatcher

nlp = spacy.load("en_core_web_sm")

# Token-based pattern matching
matcher = Matcher(nlp.vocab)

# Pattern: adjective followed by one or more nouns
pattern = [{"POS": "ADJ"}, {"POS": "NOUN", "OP": "+"}]
matcher.add("ADJ_NOUN", [pattern])

doc = nlp("The bright blue sky and cold winter morning greeted us.")
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Match: {span.text}")
# Match: bright blue sky
# Match: cold winter morning

# Phrase matching (exact phrase lookup, very fast)
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
terms = ["machine learning", "deep learning", "natural language processing"]
patterns = [nlp.make_doc(term) for term in terms]
phrase_matcher.add("TECH_TERMS", patterns)

doc = nlp("This paper covers machine learning and natural language processing.")
matches = phrase_matcher(doc)
for match_id, start, end in matches:
    print(f"Found: {doc[start:end].text}")
```

## Custom Entity Rules

```python
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")

# Add entity ruler before the NER component
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {"label": "DRUG", "pattern": "aspirin"},
    {"label": "DRUG", "pattern": [{"LOWER": "vitamin"}, {"LOWER": "d"}]},
    {"label": "DISEASE", "pattern": "diabetes"},
    {"label": "DISEASE", "pattern": [{"LOWER": "heart"}, {"LOWER": "disease"}]},
]
ruler.add_patterns(patterns)

doc = nlp("The patient takes aspirin daily for heart disease prevention.")
for ent in doc.ents:
    print(f"{ent.text:20} {ent.label_}")
# aspirin              DRUG
# heart disease        DISEASE
```

## Sentence Segmentation and Text Processing

```python
nlp = spacy.load("en_core_web_sm")
doc = nlp("Dr. Smith went to Washington. He arrived on Monday. It was cold.")

# Sentence boundaries
for sent in doc.sents:
    print(f"[{sent.start}:{sent.end}] {sent.text}")

# Lemmatization
tokens_lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

# Filter by POS
nouns = [token.text for token in doc if token.pos_ == "NOUN"]
verbs = [token.text for token in doc if token.pos_ == "VERB"]

# Similarity (requires md or lg model with vectors)
nlp_md = spacy.load("en_core_web_md")
doc1 = nlp_md("I like cats")
doc2 = nlp_md("I love dogs")
print(f"Similarity: {doc1.similarity(doc2):.3f}")
```

## Best Practices

1. Use `en_core_web_sm` for speed; use `en_core_web_trf` when accuracy matters most.
2. Process text in batches with `nlp.pipe(texts)` for better throughput.
3. Disable unused pipeline components: `nlp.select_pipes(enable=["ner"])`.
4. Use `PhraseMatcher` for exact term lookups; it is much faster than token `Matcher`.
5. Add `EntityRuler` before `ner` to give rule-based patterns priority.
6. Use `doc.to_json()` to serialize processed documents for storage.
7. For large texts, increase `nlp.max_length` or split into paragraphs first.
8. Always download the model before first use: `python3 -m spacy download <model>`.
