# Amazon Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/Transformers-Latest-yellow.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Multi-model sentiment analysis pipeline for Amazon product reviews using traditional NLP and state-of-the-art transformer models.

## Overview

This project implements and compares multiple sentiment analysis approaches on Amazon product reviews, from lexicon-based methods to cutting-edge transformer models. The analysis reveals how different models interpret sentiment and their correlation with actual star ratings.

## Features

- **Multi-Model Comparison**: VADER, RoBERTa, and DistilBERT
- **NLP Tools**: Tokenization, POS tagging, Named Entity Recognition
- **Correlation Analysis**: Sentiment scores vs. actual ratings
- **Error Analysis**: Identifying contradictory predictions
- **Visualization**: Pair plots and comparative charts

## Tech Stack

- **Python 3.8+**
- **Transformers (Hugging Face)** - Pre-trained models
- **NLTK** - Natural language processing toolkit
- **Pandas & NumPy** - Data manipulation
- **Scipy** - Statistical computations
- **Matplotlib & Seaborn** - Visualization

## Models Implemented

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type:** Lexicon-based sentiment analyzer
- **Outputs:** Negative, Neutral, Positive, Compound scores
- **Best for:** Social media text, short reviews

### 2. RoBERTa (Twitter-RoBERTa)
- **Model:** `cardiffnlp/twitter-roberta-base-sentiment`
- **Type:** Transformer-based (fine-tuned on Twitter data)
- **Outputs:** Probability distribution (negative, neutral, positive)
- **Strengths:** Context-aware, handles nuance

### 3. DistilBERT
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Type:** Distilled transformer (lighter, faster)
- **Outputs:** Binary sentiment (positive/negative)
- **Advantages:** Faster inference, smaller model size

### 4. NLTK NLP Tools
- Word tokenization
- Part-of-speech (POS) tagging
- Named Entity Recognition (NER) chunking

## Dataset

**Amazon Product Reviews**
- 500 reviews (sampled from larger dataset)
- Multi-star rating system (1-5 stars)
- Features:
  - Review text and summary
  - Star rating
  - User ID
  - Helpfulness metrics
  - Product information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shreyasnimbalkar/Amazon-review-sentiment-analysis.git
cd Amazon-review-sentiment-analysis
```

2. Install dependencies:
```bash
pip install transformers nltk scipy pandas numpy matplotlib seaborn jupyter
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook Amazon_review_sentiment_analysis.ipynb
```

## Usage

### VADER Sentiment Analysis
```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores(review_text)
# Returns: {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.8}
```

### RoBERTa Sentiment Analysis
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
# Get sentiment probabilities
```

### DistilBERT Pipeline
```python
from transformers import pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
result = sentiment_pipeline(review_text)
```

## Results & Insights

### Model Comparison
- **VADER vs RoBERTa**: RoBERTa shows better context understanding
- **Correlation with Ratings**: All models show positive correlation with star ratings
- **Error Cases**: Identified reviews where models disagree with human ratings

### Visualizations
- Pair plots comparing sentiment scores across models
- Distribution of sentiment scores by star rating
- Correlation matrices between models and actual ratings

## Key Findings

1. Transformer models (RoBERTa, DistilBERT) better capture context and sarcasm
2. VADER performs well on straightforward positive/negative reviews
3. Discrepancies often occur in mixed-sentiment reviews
4. 5-star ratings consistently show high positive sentiment across all models

## Project Structure

```
Amazon-review-sentiment-analysis/
├── Amazon_review_sentiment_analysis.ipynb
├── Reviews.csv
└── README.md
```

## Contributing

Contributions are welcome! Possible enhancements:
- Add more transformer models (BERT, GPT)
- Implement aspect-based sentiment analysis
- Create real-time sentiment API
- Add multilingual support

## License

MIT License

## Author

**Shreyas Nimbalkar**
- GitHub: [@shreyasnimbalkar](https://github.com/shreyasnimbalkar)
- Email: shreyasgnimbalkarwork@gmail.com
- LinkedIn: [Shreyas Nimbalkar](https://www.linkedin.com/in/shreyas-nimbalkar-713bb9267/)

## Acknowledgments

- Hugging Face for pre-trained transformer models
- NLTK team for comprehensive NLP tools
- Amazon Reviews dataset contributors
