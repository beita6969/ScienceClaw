---
name: nlp-analysis
description: Natural language processing for research including text mining, sentiment analysis, topic modeling, named entity recognition, text classification, and corpus analysis. Use when user needs to analyze text data, extract information from documents, do sentiment analysis, topic modeling, or text classification for research purposes. Triggers on "text mining", "sentiment analysis", "topic modeling", "NER", "named entity", "text classification", "word embeddings", "LDA", "corpus analysis", "word frequency", "TF-IDF".
---

# NLP Analysis

Text mining and NLP for scientific research. Venv: `source /Users/zhangmingda/clawd/.venv/bin/activate`

## Text Preprocessing

```python
import re
import nltk
from collections import Counter

# Download resources (first time)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess(text, lang='english'):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words(lang))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens
```

## Text Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_tfidf = tfidf.fit_transform(documents)

# Get top terms per document
feature_names = tfidf.get_feature_names_out()
for i, doc in enumerate(documents[:3]):
    scores = X_tfidf[i].toarray().flatten()
    top_idx = scores.argsort()[-10:][::-1]
    print(f"Doc {i}: {[feature_names[j] for j in top_idx]}")
```

## Topic Modeling

### LDA (Latent Dirichlet Allocation)
```python
from sklearn.decomposition import LatentDirichletAllocation

n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda.fit(X_count)  # use CountVectorizer, not TF-IDF

# Print top words per topic
for i, topic in enumerate(lda.components_):
    top_words = [feature_names[j] for j in topic.argsort()[-10:][::-1]]
    print(f"Topic {i}: {', '.join(top_words)}")

# Topic coherence: use gensim for proper evaluation
```

### BERTopic (neural topic modeling)
```python
# pip install bertopic
from bertopic import BERTopic

topic_model = BERTopic(language="english", nr_topics="auto")
topics, probs = topic_model.fit_transform(documents)
topic_model.get_topic_info()
```

## Sentiment Analysis

```python
# Simple lexicon-based
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
for text in texts:
    scores = sia.polarity_scores(text)
    print(f"{scores['compound']:.3f} | {text[:80]}")

# Transformer-based (more accurate)
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
results = sentiment(texts)
```

## Named Entity Recognition

```python
# Using transformers
from transformers import pipeline

ner = pipeline("ner", aggregation_strategy="simple")
entities = ner("Albert Einstein developed the theory of relativity at Princeton University.")
for e in entities:
    print(f"{e['entity_group']}: {e['word']} (score: {e['score']:.3f})")
```

## Text Classification

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# TF-IDF + Logistic Regression (strong baseline)
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])
scores = cross_val_score(pipe, texts, labels, cv=5, scoring='f1_macro')
print(f"F1 macro: {scores.mean():.3f} ± {scores.std():.3f}")
```

## Word Embeddings & Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Using TF-IDF vectors for document similarity
sim_matrix = cosine_similarity(X_tfidf)

# For word-level embeddings, use gensim Word2Vec or sentence-transformers
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
```

## Corpus Statistics

```python
def corpus_stats(documents):
    all_tokens = [word_tokenize(doc.lower()) for doc in documents]
    all_words = [w for tokens in all_tokens for w in tokens if w.isalpha()]
    
    print(f"Documents: {len(documents)}")
    print(f"Total tokens: {len(all_words)}")
    print(f"Vocabulary size: {len(set(all_words))}")
    print(f"Type-token ratio: {len(set(all_words))/len(all_words):.4f}")
    print(f"Avg doc length: {np.mean([len(t) for t in all_tokens]):.1f} tokens")
    
    freq = Counter(all_words)
    print(f"Top 20 words: {freq.most_common(20)}")
```

## Chinese NLP

```python
# For Chinese text, use jieba for segmentation
# pip install jieba
import jieba

text = "自然语言处理是人工智能的重要方向"
words = list(jieba.cut(text))
print(" / ".join(words))

# For Chinese sentiment/NER, use transformers with Chinese models
# e.g., bert-base-chinese, hfl/chinese-roberta-wwm-ext
```

## Tips
- Always report preprocessing steps for reproducibility
- Use multiple topic numbers and evaluate coherence
- For small datasets, TF-IDF + classical ML often beats deep learning
- Report inter-annotator agreement for labeled datasets
- Consider domain-specific stop words and vocabularies
- For Chinese text, jieba segmentation is essential
