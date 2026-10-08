# Review Sentiment Analyzer
#
# Project 1, Task 1 - Subha & Rayhan, Ironhack AI Engineering, AI FT SEPT 26.
#
# Three models are selectable: the DistilBERT the team deployed, and two TF-IDF
# models built from scratch in 01_cleaning_sentiment_clustering.ipynb. The
# TF-IDF ones score lower but are the only ones that can show which words drove
# a decision.
#
# TO ADD A TAB: append its name to TABS below and add a matching `with` block
# at the bottom. It picks up the green menu styling automatically.
#
# Run locally:  streamlit run app.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

# ----------------------------------------------------------------- models
MODELS = {
    "Complaint-finding - class_weight='balanced' - catches 58% of complaints":
        HERE / "sentiment_model.joblib",
    "Standard training - no class weighting - higher accuracy, misses 78% of complaints":
        HERE / "sentiment_model_unweighted.joblib",
}

HF_MODEL = "SebasLopez-ai/distilbert-amazon-reviews-sentiment"
HF_LABEL = "DistilBERT - fine-tuned by another author - best scores, cannot explain itself"

SCORES = {
    HF_LABEL:
        dict(accuracy=0.9184, balanced=0.7711, macro_f1=0.6863, neg_recall=0.8115),
    "Complaint-finding - class_weight='balanced' - catches 58% of complaints":
        dict(accuracy=0.8881, balanced=0.6636, macro_f1=0.5687, neg_recall=0.5802),
    "Standard training - no class weighting - higher accuracy, misses 78% of complaints":
        dict(accuracy=0.9408, balanced=0.4464, macro_f1=0.5026, neg_recall=0.2222),
}

LABELS = ["negative", "neutral", "positive"]
COLOURS = {"negative": "#e34948", "neutral": "#eda100", "positive": "#13c400"}

FLUX = "#39FF14"      # fluorescent green
FLUX_DK = "#1f9e0a"   # readable green for text on light backgrounds
INK = "#0b0b0b"

st.set_page_config(page_title="Review Sentiment Analyzer",
                   page_icon="*", layout="centered")

# ----------------------------------------------------------------- styling
st.markdown(
    f"""
    <style>
      /* buttons */
      .stButton > button, .stDownloadButton > button {{
          background: {FLUX};
          color: {INK};
          border: 2px solid {FLUX_DK};
          border-radius: 10px;
          font-weight: 600;
          letter-spacing: .2px;
          padding: .5rem 1.4rem;
          transition: filter .12s ease, transform .06s ease;
      }}
      .stButton > button:hover, .stDownloadButton > button:hover {{
          filter: brightness(1.08);
          color: {INK};
          border-color: {FLUX_DK};
      }}
      .stButton > button:active {{ transform: translateY(1px); }}
      .stButton > button:focus {{
          box-shadow: 0 0 0 3px rgba(57,255,20,.45) !important;
          color: {INK};
      }}

      /* the tab bar, used as the menu - new tabs inherit this */
      .stTabs [data-baseweb="tab-list"] {{
          gap: 8px;
          border-bottom: 2px solid rgba(57,255,20,.35);
          padding-bottom: 6px;
      }}
      .stTabs [data-baseweb="tab"] {{
          background: rgba(57,255,20,.12);
          border: 1.5px solid rgba(57,255,20,.55);
          border-radius: 999px;
          padding: 8px 20px;
          color: {FLUX_DK};
          font-weight: 600;
      }}
      .stTabs [data-baseweb="tab"]:hover {{ background: rgba(57,255,20,.22); }}
      .stTabs [aria-selected="true"] {{
          background: {FLUX} !important;
          color: {INK} !important;
          border-color: {FLUX_DK} !important;
      }}
      .stTabs [data-baseweb="tab-highlight"] {{ background: transparent; }}

      /* selection controls pick up the accent */
      .stRadio [data-baseweb="radio"] div[aria-checked="true"] {{
          background-color: {FLUX} !important;
          border-color: {FLUX_DK} !important;
      }}
      .stProgress > div > div > div > div {{ background-color: {FLUX}; }}
      h1 {{ border-bottom: 3px solid {FLUX}; padding-bottom: .3rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- loading
@st.cache_resource
def load_model(path_str):
    return joblib.load(path_str)


@st.cache_resource
def load_hf():
    """~260 MB on first use, then cached for the life of the container."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(2)
    tk = AutoTokenizer.from_pretrained(HF_MODEL)
    md = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).eval()
    return torch, tk, md


def word_contributions(text, predicted_class, vec, clf, top_n=8):
    """Which words pushed the prediction towards the chosen class.

    A linear model makes this exact: each feature's contribution is its TF-IDF
    value times its coefficient for that class. No approximation, no explainer
    library - this IS the arithmetic the model did.
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

# TO ADD A TAB: put its name here and add a `with TAB[n]:` block below.
TABS = ["Review analyzer", "About the models"]
TAB = st.tabs(TABS)

# ================================================================= tab 1
with TAB[0]:
    choice_model = st.radio(
        "Model",
        [HF_LABEL] + list(MODELS),
        help="The first is the model our team deployed. The other two we built "
             "from scratch and differ only in whether rare classes are "
             "up-weighted during fitting.",
    )

    USING_HF = choice_model == HF_LABEL
    model = vec = clf = None
    if not USING_HF:
        # 0.6 MB, instant
        model = load_model(str(MODELS[choice_model]))
        vec = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
    # DistilBERT is NOT loaded here. It is ~260 MB on a cold container, and
    # loading it at render time means every first visitor stares at a blank
    # page before the form appears. It loads on the first Analyze click.

    _s = SCORES[choice_model]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{_s['accuracy']:.4f}")
    c2.metric("Balanced acc.", f"{_s['balanced']:.4f}")
    c3.metric("Macro F1", f"{_s['macro_f1']:.4f}")
    c4.metric("Negative recall", f"{_s['neg_recall']:.4f}")

    if USING_HF:
        st.info(
            "**Best scores of the three, and it reads sarcasm.** "
            "*\"Great, it broke on day three\"* comes out negative here and "
            "positive on ours. Two caveats: this is somebody else's finished "
            "model rather than transfer learning we performed, and its training "
            "data is unpublished, so we cannot rule out that it has already seen "
            "these reviews. It also cannot show which words drove a decision."
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
        "Sarcasm (TF-IDF fails, DistilBERT does not)": "Great, it broke on day three. Exactly what I wanted.",
        "5 stars with a complaint": "Love this tablet! Only issue is the battery barely lasts a day.",
    }

    choice = st.selectbox("Try an example, or write your own", list(EXAMPLES))
    review = st.text_area(
        "Review text", value=EXAMPLES[choice], height=130,
        placeholder="The tablet is easy to use and the battery lasts a long time.")

    if st.button("Analyze sentiment"):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            if USING_HF:
                with st.spinner("Loading DistilBERT (~260 MB on first use, "
                                "then cached)..."):
                    _torch, _tk, _md = load_hf()
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

            # A 52/22/27 split is the model saying "I don't know". Printing that
            # as a confident label, in the same red as a 98% call, is the most
            # common way a demo of this kind misleads its audience.
            CONFIDENT = 0.60

            st.markdown("### Prediction")
            if confidence < CONFIDENT:
                st.markdown(
                    "<span style='font-size:2.2rem;font-weight:700;color:#898781'>"
                    "uncertain</span>", unsafe_allow_html=True)
                st.caption(
                    f"Closest label is **{pred}**, but only at {confidence:.0%}. "
                    f"The three classes are too close to call this one.")
            else:
                st.markdown(
                    f"<span style='font-size:2.2rem;font-weight:700;"
                    f"color:{COLOURS[pred]}'>{pred}</span>",
                    unsafe_allow_html=True)

            cols = st.columns(3)
            for col, lab in zip(cols, LABELS):
                col.metric(lab.capitalize(), f"{proba[order.index(lab)]:.1%}")
            st.progress(float(proba[order.index(pred)]))

            st.markdown("### Why")
            if USING_HF:
                st.info(
                    "A transformer cannot show this. Its decision is spread "
                    "across 66 million weights with no named features to "
                    "attribute it to. Switch to either TF-IDF model to see the "
                    "per-word arithmetic.")
            else:
                contrib = word_contributions(review, pred, vec, clf)
                if contrib.empty:
                    st.info("No recognised words — every term was unseen or a stopword.")
                else:
                    st.caption(
                        f"Each word's TF-IDF weight times its coefficient for "
                        f"**{pred}**. This is the arithmetic the model performed, "
                        f"not an approximation of it.")

                    # Spell the direction out. A bar pointing down under a red
                    # "negative" heading reads as "this word is negative" when it
                    # means the exact opposite.
                    shown = contrib.copy()
                    shown["effect"] = [
                        ("pushes TOWARD " + pred) if v > 0
                        else ("pushes AWAY from " + pred)
                        for v in shown["contribution"]
                    ]
                    st.dataframe(
                        shown[["word", "effect", "contribution"]]
                        .style.format({"contribution": "{:+.3f}"}),
                        hide_index=True, use_container_width=True)

                    defenders = contrib[contrib["contribution"] < 0]["word"].tolist()
                    if defenders and pred == "negative":
                        st.caption(
                            "Arguing against this verdict: **"
                            + "**, **".join(defenders[:4])
                            + "**. They were outweighed, not ignored.")

            if confidence < CONFIDENT and not USING_HF:
                st.warning(
                    "**Why this is uncertain.** These TF-IDF models only know "
                    "Amazon tablets, e-readers and Echo devices. Phrases like "
                    "*\"worked as expected\"* lean negative in this data, because "
                    "reviewers write them as *\"worked for two weeks\"*. Words "
                    "from outside the domain have coefficients fitted to a "
                    "handful of reviews. Try DistilBERT above on the same text.")

# ================================================================= tab 2
with TAB[1]:
    st.markdown(
        """
### Three models, same test set

| model | accuracy | balanced acc. | macro F1 | negative recall |
|---|---|---|---|---|
| **DistilBERT** (deployed default) | 0.9184 | **0.7711** | **0.6863** | **0.8115** |
| TF-IDF, `class_weight="balanced"` | 0.8881 | 0.6636 | 0.5687 | 0.5802 |
| TF-IDF, unweighted | **0.9408** | 0.4464 | 0.5026 | 0.2222 |

Held out: 6,926 reviews, 20% stratified split, seed 42.

### Why accuracy is not the headline

The dataset is **93.3% positive**, so answering "positive" every time already scores
93.3%. Multinomial Naive Bayes on these features reaches 93.3% accuracy while predicting
**zero** negative reviews — none of the 162 genuine negatives in the test set.

Selection was therefore on **negative recall and macro F1**, which picks a model with
*lower* accuracy. For a system meant to surface complaints, a missed complaint is the
expensive error.

### What the TF-IDF models are built from

| | |
|---|---|
| features | review **title + body**, TF-IDF, unigrams + bigrams, 20,000 terms |
| classifier | logistic regression, `max_iter=2000` |
| trained on | 34,626 cleaned reviews |
| model file | **0.6 MB**, predicts in milliseconds |

`class_weight="balanced"` weights the negative class about **14×** against positive's
0.36×. That takes negative recall from 0.13 to 0.58 — and is also why mild praise like
*"good"* reads as neutral, since neutral is boosted 7.7×.

### Known limits

- **Neutral is the hardest class** and the most often confused.
- **TF-IDF misreads sarcasm.** *"Great, it broke on day three"* → positive. DistilBERT
  gets it right; bag-of-words cannot see a contradiction across a comma.
- **Star rating is a proxy** for sentiment, not sentiment. A 5-star review containing a
  complaint is labelled positive in training.
- **DistilBERT's training data is unpublished.** If it overlaps this dataset it has
  effectively seen the test set, and 0.8115 negative recall on a class that is 2.3% of
  the data is exactly what that would look like.
        """
    )
