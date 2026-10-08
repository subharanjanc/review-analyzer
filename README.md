# Review Sentiment Analyzer

Live app: **https://subha-review-analyzer.streamlit.app**

Task 1 of [Project 1 — NLP Automated Customer Reviews](https://github.com/rayhanpatoary/CustomerReviewproject),
Ironhack AI Engineering, cohort AI FT SEPT 26. **Subha & Rayhan.**

Classifies an Amazon product review as negative, neutral or positive — and shows **which
words drove the decision**.

---

## The model

Built from scratch in `01_cleaning_sentiment_clustering.ipynb`. No transformer.

| | |
|---|---|
| features | review **title + body**, TF-IDF, unigrams + bigrams, 20,000 terms, `min_df=2`, English stopwords |
| classifier | `LogisticRegression(class_weight="balanced", max_iter=2000)` |
| trained on | 34,626 cleaned reviews |
| model file | **0.6 MB** |
| prediction | milliseconds, CPU |

### Held-out results

6,926 reviews, 20% stratified split, seed 42:

| metric | value |
|---|---|
| accuracy | 0.8881 |
| balanced accuracy | 0.6636 |
| macro F1 | 0.5687 |
| **negative recall** | **0.580** |

---

## Why accuracy is not the headline

The dataset is **93.3% positive**. Answering "positive" every time already scores 93.3%.

On the same features, Multinomial Naive Bayes reaches **93.3% accuracy while predicting
zero negative reviews** — not one, across all 162 genuine negatives in the test set. It
learned the class distribution, not the language.

So selection was on **negative recall and macro F1**, and the model chosen has *lower*
accuracy than three of the alternatives. For a system meant to surface complaints, a
missed complaint is the expensive error.

`class_weight="balanced"` is what buys that: it weights the negative class about **14×**
against positive's 0.36×, taking negative recall from 0.13 to 0.58 for seven points of
accuracy.

---

## Why a linear model, deployed

| | this app | a DistilBERT app |
|---|---|---|
| model size | **0.6 MB** | ~260 MB download |
| dependencies | scikit-learn | torch + transformers |
| first prediction | **milliseconds** | several seconds, cold |
| explains itself | **yes, exactly** | no |

That last row is the real argument. For a linear model the explanation is not an
approximation — each word's contribution is literally

```python
contribution = tfidf_weight * coefficient_for_predicted_class
```

which **is** the arithmetic the model performed. No LIME, no SHAP. The app shows it for
every prediction.

A fine-tuned DistilBERT does score higher on this data (macro F1 0.6327 against 0.5687).
It cannot tell you why.

---

## Known limits

- **Neutral is the hardest class** — recall 0.497, and the one most often confused.
- **Sarcasm is usually misread.** Try *"Great, it broke on day three."*
- **Star rating is a proxy for sentiment**, not sentiment. A 5-star review containing a
  complaint is labelled positive in training.
- **42 products, 83% one brand.** This will not transfer to a general review corpus.

The app ships five examples, two of them chosen to fail honestly.

---

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```
app.py                   the Streamlit app
sentiment_model.joblib   the fitted scikit-learn pipeline
requirements.txt         UTF-8, no BOM
```
