# Amazon review sentiment analysis

A notebook that scores Amazon product reviews with two sentiment models, VADER and
RoBERTa, and checks how their scores line up with the star rating each reviewer gave.

## Data

The notebook reads a `Reviews.csv` file of Amazon reviews (columns include `Score`
for the 1 to 5 star rating, `Summary` and `Text`) and uses the first 500 rows. The CSV
is not in the repo; in Colab the notebook expects it at `/content/Reviews.csv`.

## What the notebook does

1. Plots how many reviews there are for each star rating.
2. Runs basic NLTK steps on one review: tokenizing, part-of-speech tagging and named
   entity chunking.
3. Scores every review with VADER (NLTK's rule-based sentiment lexicon), which
   gives negative, neutral, positive and compound scores.
4. Scores every review with RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`
   from Hugging Face), which gives negative, neutral and positive probabilities.
5. Plots the six scores against each other, colored by star rating.
6. Looks at the reviews where the models and the stars disagree most.
7. Tries the default Hugging Face `sentiment-analysis` pipeline (DistilBERT fine-tuned
   on SST-2) on a few short example sentences.

## What I found

- On a clearly negative review ("This oatmeal is not good..."), RoBERTa gave 0.976
  negative.
- RoBERTa failed on 2 of the 500 reviews (ids 83 and 187), so 498 reviews were scored
  by both models.
- The disagreements are the interesting part. The 1-star review VADER rated most
  positive is "So we cancelled the order. It was cancelled without any problem. That is
  a positive note...": VADER picks up the positive words and misses the context.
- Both models gave the highest negative score among 5-star reviews to the same text:
  "this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my
  fault". It's a happy review written with negative words.

I didn't compute an accuracy or correlation number, so this is an exploration rather
than a benchmark. The DistilBERT pipeline was only tried on toy sentences, not on the
reviews.

## How to run

The notebook was written for Google Colab. Open it with the Colab badge at the top of
`Amazon_review_sentiment_analysis.ipynb`, upload `Reviews.csv` to `/content/`, and
run all cells. It installs NLTK and downloads the NLTK data it needs.

To run it locally, change the CSV path and install:

```bash
pip install pandas numpy matplotlib seaborn nltk transformers torch scipy tqdm jupyter
```

## Files

- `Amazon_review_sentiment_analysis.ipynb`: the main notebook
- `Amazon_review_sentiment_analysis (1).ipynb`: an earlier copy of the same notebook
  without the Colab badge
- `LICENSE`: MIT
