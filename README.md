# Real or Not? NLP Disaster Tweet Classification

A supervised NLP project that classifies tweets as being about a real disaster (`target = 1`) or not (`target = 0`), built on the Kaggle ["Real or Not? NLP with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started) dataset.

The notebook is written as a transparent, end-to-end case study: every score reported was actually measured, including experiments that didn't help. It walks through EDA, a proposed modeling plan, a critique of that plan, thirteen modeling iterations, a 5-fold cross-validation robustness check, and an honest discussion of the realistic performance ceiling for this dataset — rather than optimizing toward a cherry-picked number.

**Final result: ~81.5% accuracy / ~77.1% F1** (5-fold cross-validation, ±1 percentage point across folds).

## Contents

- [`disaster_tweets_classification.ipynb`](disaster_tweets_classification.ipynb) — the full analysis and modeling notebook
- `train.csv`, `test.csv`, `sample_submission.csv` — the Kaggle competition data
- `submission.csv` — predictions produced by the final model, in the competition's submission format

## Approach

1. **Exploratory Data Analysis** — class balance, label noise (duplicate tweets with contradictory labels), text length, URLs/mentions/hashtags, the `keyword` field, and the `location` field.
2. **Train / validation split** — a stratified 85/15 split, used consistently across every iteration to keep comparisons fair.
3. **Modeling outline, then a critique of that outline** — a plan is proposed first, then reviewed critically before any code is written, to surface risks like relying on a single validation split or anchoring to an arbitrary accuracy target.
4. **Text cleaning & feature engineering** — mojibake/HTML-entity cleanup, URL/mention stripping, hashtag-word preservation, hand-crafted meta-features (length, punctuation counts, URL/mention/hashtag counts), and keyword handling.
5. **Iterative model development** — starting from a TF-IDF + Logistic Regression baseline and progressively adding: alternative classical algorithms (Naive Bayes, Linear SVM), meta-features, the `keyword` field (one-hot, then smoothed target-encoding), hyperparameter tuning, character n-grams, ensembling (soft voting, then stacking), a tree-based model (XGBoost) for comparison, an empirical test of removing label-noise rows, and an optional pretrained-embeddings (GloVe-Twitter) experiment.
6. **5-fold cross-validation** — the winning architecture is re-evaluated with proper cross-validation (refitting every vectorizer/encoder per fold) to get a reliable, split-independent performance estimate rather than trusting a single held-out set.
7. **Honest discussion of the score ceiling** — label noise, inherent metaphor/literal ambiguity in short tweets, and a modest training set all cap what classical NLP methods can achieve here; the notebook explains why ~81-85% is the realistic range for this dataset (matching published benchmarks) rather than treating a lower-than-hoped score as a failure.
8. **Final model** — the winning architecture retrained on 100% of `train.csv`, used to generate predictions on `test.csv`.

## Final model architecture

A **stacking ensemble** — Logistic Regression, a calibrated Linear SVM, and Complement Naive Bayes as base learners, combined by a Logistic Regression meta-learner — over:

- Word-level TF-IDF (1-2 grams, 30,000 features)
- Character-level TF-IDF (3-5 grams, 30,000 features, `char_wb` analyzer)
- Hand-crafted meta-features (text length, punctuation/URL/mention/hashtag counts, etc.)
- A smoothed target-encoded `keyword` field

## What didn't help (and is reported anyway)

- **XGBoost** underperformed the linear models — tree-based models split on one feature at a time, which fits poorly with extremely high-dimensional, sparse TF-IDF representations.
- **Removing label-noise rows** (tweets with contradictory labels) from training didn't move validation scores — they're a small fraction of the data, and similar ambiguity persists in the eval set regardless.
- **Averaged pretrained GloVe-Twitter embeddings** underperformed TF-IDF — averaging discards word order and can't distinguish literal from metaphorical usage, which is the central difficulty of this dataset.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost jupyter
jupyter notebook disaster_tweets_classification.ipynb
```

Run all cells top to bottom. The optional pretrained-embeddings experiment (Section 7.11) is disabled by default since it downloads ~390 MB of GloVe-Twitter vectors; set `RUN_GLOVE_EXPERIMENT = True` in that cell to reproduce it (requires internet access).

**Tested with:** Python 3.14, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18, xgboost 3.4.

## Limitations & future work

- No transformer fine-tuning (BERT/RoBERTa) was attempted; this is the most promising direction for further gains.
- `location` was dropped as a feature (too messy/free-text) rather than cleaned or geocoded.
- The `keyword` target-encoding smoothing constant is fixed rather than tuned.
- A domain-specific Word2Vec/FastText model trained on this corpus (instead of generic pretrained vectors) is untried.

## License

The dataset is from the Kaggle ["Real or Not? NLP with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started) competition; see Kaggle for its terms of use. Code in this repository is provided for educational purposes.
