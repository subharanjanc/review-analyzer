# Review Sentiment Analyzer
#
# Task 1 of Project 1, deployed. The model is the one built from scratch in
# 01_cleaning_sentiment_clustering.ipynb: TF-IDF over review title + body,
# then logistic regression with class_weight="balanced".
#
# No transformer here, deliberately. The whole model is 0.6 MB and predicts in
# milliseconds, where a DistilBERT pipeline needs a ~260 MB download and
# several seconds per cold start. It is also the only one of the two that can
# show WHY it decided what it decided - see the word contributions below.
#
# Run locally:  streamlit run app.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Cloud runs with the REPO ROOT as the working directory, not the
# folder holding this file - so a bare "sentiment_model.joblib" would not be
# found once deployed. Resolve it against this file instead.
HERE = Path(__file__).parent

# Two models, same features, differing only in class_weight. Keeping both
# makes the central trade-off of this project something you can switch on and
# off rather than something you have to take on trust.
MODELS = {
    "Complaint-finding - class_weight='balanced' - catches 58% of complaints": HERE / "sentiment_model.joblib",
    "Standard training - no class weighting - higher accuracy, misses 78% of complaints": HERE / "sentiment_model_unweighted.joblib",
}

# A third option: the off-the-shelf model Rayhan's dashboard uses. It beats
# ours on every metric and handles sarcasm, which ours misses. It is kept as an
# option rather than a replacement because (a) it is somebody else's finished
# model, not transfer learning we performed, (b) its training data is
# unpublished so we cannot rule out that it has seen these reviews, and (c) it
# cannot show which words drove a decision.
HF_MODEL = "SebasLopez-ai/distilbert-amazon-reviews-sentiment"
HF_LABEL = "DistilBERT - fine-tuned by another author - best scores, cannot explain itself"

SCORES = {
    "Complaint-finding - class_weight='balanced' - catches 58% of complaints":
        dict(accuracy=0.8881, balanced=0.6636, macro_f1=0.5687, neg_recall=0.5802),
    "Standard training - no class weighting - higher accuracy, misses 78% of complaints":
        dict(accuracy=0.9408, balanced=0.4464, macro_f1=0.5026, neg_recall=0.2222),
    HF_LABEL:
        dict(accuracy=0.9184, balanced=0.7711, macro_f1=0.6863, neg_recall=0.8115),
}

st.set_page_config(page_title="Review Sentiment Analyzer",
                   page_icon="*", layout="centered")

LABELS = ["negative", "neutral", "positive"]
COLOURS = {"negative": "#e34948", "neutral": "#eda100", "positive": "#1baf7a"}


@st.cache_resource
def load_model(path_str):
    return joblib.load(path_str)


def word_contributions(text, predicted_class, top_n=8):
    """Which words pushed the prediction towards the chosen class.

    A linear model makes this exact: each feature's contribution is simply its
    TF-IDF value times its coefficient for that class. No approximation, no
    explainer library - this IS the arithmetic the model did.
    """
    x = vec.transform([text])
    if x.nnz == 0:
        return pd.DataFrame(columns=["word", "contribution"])

    names = np.asarray(vec.get_feature_names_out())
    k = list(clf.classes_).index(predicted_class)
    idx = x.nonzero()[1]
    contrib = x.toarray()[0][idx] * clf.coef_[k][idx]

    df = pd.DataFrame({"word": names[idx], "contribution": contrib})
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
    return df.head(top_n).reset_index(drop=True)


st.title("Review Sentiment Analyzer")
st.caption("Project 1 · Task 1 · Subha & Rayhan — Ironhack AI Engineering, AI FT SEPT 26")

st.write(
    "Paste an Amazon product review and get a sentiment classification. "
    "Three models are available: the DistilBERT our team deployed, and the "
    "two TF-IDF models we built from scratch - which score lower but can "
    "show exactly which words drove the decision."
)

@st.cache_resource
def load_hf():
    """~260 MB on first use, then cached for the life of the container."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(2)
    tk = AutoTokenizer.from_pretrained(HF_MODEL)
    md = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).eval()
    return torch, tk, md


choice_model = st.radio(
    "Model",
    [HF_LABEL] + list(MODELS),
    horizontal=False,
    help="Same features and same training data. The only difference is whether "
         "the rare classes are up-weighted during fitting.",
)

USING_HF = choice_model == HF_LABEL
if USING_HF:
    with st.spinner("Loading DistilBERT (~260 MB on first use)..."):
        _torch, _tk, _md = load_hf()
    model = vec = clf = None
else:
    model = load_model(str(MODELS[choice_model]))
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

_s = SCORES[choice_model]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{_s['accuracy']:.4f}")
c2.metric("Balanced acc.", f"{_s['balanced']:.4f}")
c3.metric("Macro F1", f"{_s['macro_f1']:.4f}")
c4.metric("Negative recall", f"{_s['neg_recall']:.4f}")

if USING_HF:
    st.info(
        "**Best scores of the three, and it reads sarcasm** - "
        "*\"Great, it broke on day three\"* comes out negative here and "
        "positive on ours. Two caveats worth stating: this is somebody else's "
        "finished model rather than transfer learning we performed, and its "
        "training data is unpublished, so we cannot rule out that it has "
        "already seen these reviews. It also cannot show which words drove a "
        "decision - the table below is unavailable for it."
    )
elif choice_model.startswith("Standard"):
    st.warning(
        "**0.9408 accuracy, and it finds 22% of complaints.** The balanced "
        "model scores 0.8881 and finds 58%. Accuracy is higher here only "
        "because 93.3% of reviews are positive, so leaning positive is right "
        "most of the time. Fine for judging one review; wrong for a system "
        "whose job is surfacing unhappy customers."
    )

st.divider()

EXAMPLES = {
    "— choose an example —": "",
    "Clear complaint": "Battery died after two weeks and support never replied. Waste of money.",
    "Clear praise": "Easy to set up and the screen is lovely. My daughter uses it every day.",
    "Genuinely ambiguous": "It's fine, I guess. Does what it says. Nothing special.",
    "Sarcasm (models usually fail)": "Great, it broke on day three. Exactly what I wanted.",
    "5 stars with a complaint": "Love this tablet! Only issue is the battery barely lasts a day.",
}

choice = st.selectbox("Try an example, or write your own", list(EXAMPLES))
review = st.text_area("Review text", value=EXAMPLES[choice], height=130,
                      placeholder="The tablet is easy to use and the battery lasts a long time.")

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Enter a review first.")
    else:
        if USING_HF:
            enc = _tk(review, return_tensors="pt", truncation=True, max_length=256)
            with _torch.no_grad():
                logits = _md(**enc).logits[0]
            proba = _torch.softmax(logits, -1).numpy()
            order = [_md.config.id2label[i].lower() for i in range(len(proba))]
        else:
            proba = model.predict_proba([review])[0]
            order = list(clf.classes_)
        pred = order[int(np.argmax(proba))]
        confidence = float(proba.max())

        # A 39% / 31% / 30% split is the model saying "I don't know". Showing
        # that as a confident label - in the same red as a 98% call - is
        # misleading, and it is the single most common way a demo of this kind
        # misleads its audience. Below CONFIDENT the label is withheld.
        CONFIDENT = 0.60

        st.markdown("### Prediction")
        if confidence < CONFIDENT:
            st.markdown(
                "<span style='font-size:2rem;font-weight:600;color:#898781'>"
                "uncertain</span>", unsafe_allow_html=True)
            st.caption(
                f"Closest label is **{pred}**, but only at {confidence:.0%}. "
                f"The three classes are too close to call this one."
            )
        else:
            st.markdown(
                f"<span style='font-size:2rem;font-weight:600;color:{COLOURS[pred]}'>"
                f"{pred}</span>", unsafe_allow_html=True)

        cols = st.columns(3)
        for col, lab in zip(cols, LABELS):
            col.metric(lab.capitalize(), f"{proba[order.index(lab)]:.1%}")

        st.progress(float(proba[order.index(pred)]))

        st.markdown("### Why")
        if USING_HF:
            st.info(
                "A transformer cannot show this. Its decision is spread across "
                "66 million weights with no named features to attribute it to. "
                "Switch to either TF-IDF model to see the per-word arithmetic."
            )
            st.stop()
        contrib = word_contributions(review, pred)
        if contrib.empty:
            st.info("No recognised words — every term was unseen or a stopword.")
        else:
            st.caption(
                f"Each word's TF-IDF weight times its coefficient for "
                f"**{pred}**. This is the arithmetic the model performed, not "
                f"an approximation of it."
            )

            # Spell the direction out in words. A bar pointing down under a red
            # "negative" heading reads as "this word is negative" when it means
            # the exact opposite, and that has already misled a reader once.
            shown = contrib.copy()
            shown["effect"] = [
                ("pushes TOWARD " + pred) if v > 0 else ("pushes AWAY from " + pred)
                for v in shown["contribution"]
            ]
            shown = shown[["word", "effect", "contribution"]]
            st.dataframe(shown.style.format({"contribution": "{:+.3f}"}),
                         hide_index=True, use_container_width=True)

            defenders = contrib[contrib["contribution"] < 0]["word"].tolist()
            if defenders and pred == "negative":
                st.caption(
                    "Arguing against this verdict: **"
                    + "**, **".join(defenders[:4])
                    + "**. They were outweighed, not ignored."
                )

        if confidence < CONFIDENT:
            st.warning(
                "**Why this is uncertain.** The model only knows Amazon tablets, "
                "e-readers and Echo devices. Phrases like *\"worked as expected\"* "
                "lean negative in this data, because reviewers write them as "
                "*\"worked for two weeks\"* or *\"what I expected for the price\"*. "
                "Words from outside the domain — smartphone, laptop, headphones — "
                "have coefficients fitted to a handful of reviews and should not "
                "have coefficients fitted to a handful of reviews and should "
                "not be trusted. The chart above shows exactly which words "
                "drove it. Try the **unweighted** model above on the same "
                "text - mild praise like 'good' reads as positive there, "
                "because neutral is no longer multiplied by 7.7x."
            )

with st.expander("About this model"):
    st.markdown(
        """
**TF-IDF + logistic regression**, trained from scratch on 34,626 cleaned Amazon reviews.

| | |
|---|---|
| features | review **title + body**, TF-IDF, unigrams + bigrams, 20,000 terms |
| classifier | logistic regression, `class_weight="balanced"` |
| size | **0.6 MB** |
| prediction time | milliseconds, CPU |

**Held-out test results** (6,926 reviews, 20% split):

| metric | value |
|---|---|
| accuracy | 0.8881 |
| balanced accuracy | 0.6636 |
| macro F1 | 0.5687 |
| **negative recall** | **0.580** |

**Why accuracy is not the headline.** The dataset is 93.3% positive, so answering
"positive" every time already scores 93.3%. A Naive Bayes model on the same features
reaches 93.3% accuracy while predicting **zero** negative reviews. We selected on
negative recall and macro F1 instead, and chose the model with *lower* accuracy —
for a system meant to surface complaints, a missed complaint is the expensive error.

**Known limits.** Neutral is the hardest class (recall 0.497) and the one most often
confused. Sarcasm is usually misread. Star rating is a *proxy* for sentiment, so a
5-star review containing a complaint is labelled positive in training.
        """
    )
