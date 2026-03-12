---
name: nltk-linguistics
description: "NLP and corpus analysis via NLTK. Use when: user asks about tokenization, POS tagging, parsing, sentiment, or corpus statistics. NOT for: deep learning NLP or transformer models."
metadata: { "openclaw": { "emoji": "🗣️", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-nltk", "kind": "uv", "package": "nltk" }] } }
---

# NLTK Linguistics

Natural language processing and corpus analysis using NLTK.

## Setup

```python
import nltk
for pkg in ['punkt_tab', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker_tab',
            'words', 'vader_lexicon', 'wordnet', 'stopwords']:
    nltk.download(pkg, quiet=True)
```

## Tokenization

```python
from nltk.tokenize import word_tokenize, sent_tokenize
sentences = sent_tokenize(text)
words = word_tokenize(text)
```

## POS Tagging

```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize
tagged = pos_tag(word_tokenize(text))  # list of (word, tag) tuples
# Tags: NN=noun, VB=verb, JJ=adjective, RB=adverb, DT=determiner
```

## Named Entity Recognition

```python
from nltk import ne_chunk, pos_tag, word_tokenize
tree = ne_chunk(pos_tag(word_tokenize(text)))
for subtree in tree:
    if hasattr(subtree, 'label'):
        entity = " ".join(word for word, tag in subtree.leaves())
        print(f"{subtree.label()}: {entity}")
```

## Sentiment Analysis (VADER)

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores(text)
# Returns: {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.6369}
# compound: -1 (most negative) to +1 (most positive)
```

## Frequency Distributions and Concordance

```python
from nltk import FreqDist, Text
from nltk.tokenize import word_tokenize
fdist = FreqDist(word_tokenize(text.lower()))
fdist.most_common(20)                       # top 20 words
t = Text(word_tokenize(text))
t.concordance('language', width=80)         # keyword-in-context
t.collocations()                            # frequent bigrams
```

## WordNet Lookups

```python
from nltk.corpus import wordnet as wn
synsets = wn.synsets('bank')                # all senses
defn = synsets[0].definition()              # definition string
sim = wn.synset('dog.n.01').wup_similarity(wn.synset('cat.n.01'))  # Wu-Palmer similarity
synonyms = [l.name() for s in wn.synsets('good') for l in s.lemmas()]
hypernyms = wn.synset('dog.n.01').hypernyms()
```

## Stopword Filtering

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]
```

## Best Practices

1. Always download required NLTK data before first use.
2. Use `word_tokenize` over `split()` for proper tokenization.
3. VADER works best on short social-media-style text.
4. For large corpora, consider streaming with `PlaintextCorpusReader`.
5. POS tag sets: use `nltk.help.upenn_tagset()` for tag reference.
6. WordNet similarity requires both synsets to share a common hypernym.
