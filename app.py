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


EMB_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


@st.cache_resource
def load_embedder():
    """MiniLM-L12, 33M parameters. Embeds a product name in well under a second."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(2)
    tk = AutoTokenizer.from_pretrained(EMB_MODEL)
    md = AutoModel.from_pretrained(EMB_MODEL).eval()
    return torch, tk, md


@st.cache_data
def load_clusters():
    import json
    cent = np.load(HERE / "cluster_centroids.npy")
    with open(HERE / "cluster_names.json", encoding="utf-8") as f:
        names = {int(k): v for k, v in json.load(f).items()}
    prods = pd.read_csv(HERE / "products_categorised.csv")
    return cent, names, prods


@st.cache_data
def load_catstring_clusters():
    """The second model: clusters Amazon's own `categories` strings instead of
    product names. Neither ratings nor review counts are used as features."""
    import json
    cent = np.load(HERE / "catstring_centroids.npy")
    with open(HERE / "catstring_names.json", encoding="utf-8") as f:
        names = {int(k): v for k, v in json.load(f).items()}
    tbl = pd.read_csv(HERE / "catstring_table.csv")
    return cent, names, tbl


@st.cache_data
def load_members(using_cat):
    """Embeddings of the real items the clustering was fitted on, in the same
    row order as products_categorised.csv / catstring_table.csv.

    Used only to decide whether an input resembles anything in the dataset at
    all - see the floor test in the Product categories tab. Unit vectors, so a
    dot product is cosine similarity. 39x384 and 38x384 floats, ~60 KB each.
    """
    f = "catstring_embeddings.npy" if using_cat else "product_embeddings.npy"
    return np.load(HERE / f)


@st.cache_data
def load_summaries():
    """Task 3 articles, precomputed by notebooks/task3_hybrid_summaries.ipynb.

    Structured rather than flat: the notebook writes one paragraph per
    category for Rayhan's dashboard, this file keeps the parts separate so the
    tab can lay them out. Same numbers, same complaint mining, same quotes,
    same generated sentence - nothing is recomputed here, and no model runs at
    request time.
    """
    import json
    with open(HERE / "category_summaries_structured.json", encoding="utf-8") as f:
        return json.load(f)


def build_flat(cat, s):
    """Reassemble the notebook's one-paragraph article from the parts.

    Kept identical to build_article() in the notebook so the expander shows
    exactly what category_summaries.json holds - the format Rayhan's dashboard
    reads.
    """
    L = ["%s - %s reviews across %d products, averaging %.2f stars."
         % (cat, format(s["reviews"], ","), s["n_products"], s["rating"]),
         "%.1f%% of reviews are positive and %.1f%% negative."
         % (s["pct_positive"], s["pct_negative"]),
         "Most reviewed: " + "; ".join(
             "%s (%s reviews)" % (p["name"][:60], format(p["reviews"], ","))
             for p in s["top3"]) + "."]

    w = s["worst"]
    if w is not None and w["stands_out"]:
        L.append("Most complained about: %s - %.1f%% of its %s reviews are "
                 "negative, against %.1f%% for the category, at %.2f stars."
                 % (w["name"][:60], w["neg_rate"], format(w["reviews"], ","),
                    s["pct_negative"], w["rating"]))
    elif w is not None:
        L.append("No single product stands out as worse than the rest; the "
                 "highest negative rate is %.1f%% against %.1f%% for the "
                 "category." % (w["neg_rate"], s["pct_negative"]))

    if s["complaints"]:
        L.append("Complaints centre on: " + ", ".join(
            "%s (in %d reviews)" % (c["word"], c["n_reviews"])
            for c in s["complaints"]) + ".")
    else:
        L.append("Too few negative reviews here to identify complaint themes.")

    if s["generated"]:
        L.append("What unhappy customers say: " + s["generated"])
    elif s["quotes"]:
        q = s["quotes"][0]
        L.append("The most endorsed complaint (%d readers found it helpful): "
                 "\"%s\"" % (q["helpful"], q["text"][:220]))

    return " ".join(L)


def embed_one(text):
    torch, tk, md = load_embedder()
    enc = tk([text], padding=True, truncation=True, max_length=64,
             return_tensors="pt")
    with torch.no_grad():
        h = md(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    v = ((h * m).sum(1) / m.sum(1)).numpy()
    return (v / np.linalg.norm(v, axis=1, keepdims=True))[0]


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
TABS = ["Review analyzer", "Product categories", "Category summaries",
        "About the models"]
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

            # A 52/22/27 split is the model saying "I don't know". Printing
            # that as a confident label, in the same red as a 98% call, is the
            # most common way a demo of this kind misleads its audience.
            #
            # The threshold is per-model because the two kinds are calibrated
            # differently. TF-IDF regularly lands near a three-way tie and needs
            # the guard. DistilBERT is sharper - across ten probe sentences it
            # dropped below 60% once - so holding it to the same bar would
            # withhold labels Rayhan's dashboard prints, and the two of us would
            # demo different answers from the same weights.
            # 0.40 for DistilBERT: with three classes the floor is 33%, so
            # this only withholds a label when the output is close to random.
            # It matches Rayhan's dashboard on every sentence we have tested.
            CONFIDENT = 0.40 if USING_HF else 0.60

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
    st.subheader("Which category does a product belong to?")
    st.write(
        "Task 2. The 39 products were grouped into six meta-categories by "
        "embedding each **product name** with a sentence model and running "
        "KMeans over those vectors. Type any product name below and it is "
        "placed in the nearest category - **or told that none of the six "
        "fits**, if nothing in the dataset resembles it. Try *Samsung Galaxy "
        "S9*: there is no phone cluster to put it in, and saying so is the "
        "correct answer."
    )

    BY_NAME = "By product name - 39 products, ARI 0.9999, covers 80.5% of reviews"
    BY_CAT = "By Amazon category string - 38 strings, ARI 0.8107, covers 100%"
    which = st.radio("Clustering model", [BY_NAME, BY_CAT], key="cluster_model")
    USING_CAT = which == BY_CAT

    if USING_CAT:
        cent, cnames, tbl = load_catstring_clusters()
        st.info(
            "**Clusters Amazon's own `categories` field rather than the product "
            "name.** Neither ratings nor review counts are used. It scores lower "
            "against the hand-labelled answer (0.8107 vs 0.9999) but reaches "
            "**every** review — including the **6,759 with no product name at "
            "all**, which the other model cannot touch. Those turn out to be "
            "TV and home-theatre products: their category string says so even "
            "though their name is empty. "
            "The 0.9999 is also flattering: the gold labels are derived "
            "from product names, so a model clustering product names is "
            "largely grading its own homework. This one never sees a "
            "product name and still recovers the same taxonomy."
        )
        st.caption(
            "A caution on this model: these strings are Amazon **merchandising "
            "paths**, not a product taxonomy. One of them literally begins "
            "*Walmart for Business*, another *Back To College*. The clusters "
            "therefore group by where a product is sold as much as by what it "
            "is, and an invented breadcrumb will often land in the wrong one. "
            "The examples below are real strings taken from the data."
        )
    else:
        cent, cnames, prods = load_clusters()

    PRODUCT_EXAMPLES = {
        "— choose an example —": "",
        "A tablet": "Fire HD 10 Tablet, 32 GB, Wi-Fi",
        "An e-reader": "Kindle Paperwhite e-reader with backlight",
        "A charger": "Official 9W USB power adapter",
        "A case": "Leather protective cover for Kindle",
        "A speaker": "Echo Dot",
        "A phone - not in the dataset": "Samsung Galaxy S9",
        "Something it has never seen": "Bluetooth running headphones",
    }
    # Real strings from the dataset, one per cluster. Invented breadcrumbs sit
    # outside the training distribution and misclassify - these strings are
    # Amazon merchandising paths, not a product taxonomy, so a plausible-looking
    # made-up one ("Kindle Store, Kindle E-readers") lands in the wrong cluster.
    import json as _json
    with open(HERE / "catstring_examples.json", encoding="utf-8") as _f:
        _real = _json.load(_f)
    CAT_EXAMPLES = {"— choose a real category string —": ""}
    CAT_EXAMPLES.update({k: v for k, v in _real.items()})
    ex = CAT_EXAMPLES if USING_CAT else PRODUCT_EXAMPLES
    pick = st.selectbox("Try an example, or write your own", list(ex),
                        key="cat_example" if USING_CAT else "prod_example")
    product = st.text_input(
        "Amazon category string" if USING_CAT else "Product name",
        value=ex[pick],
        placeholder=("Fire Tablets,Tablets,Computers & Tablets" if USING_CAT
                     else "Fire HD 8 Tablet, 16 GB, Wi-Fi"))

    if st.button("Categorise product"):
        if not product.strip():
            st.warning("Enter a product name first.")
        else:
            with st.spinner("Embedding..."):
                v = embed_one(product)
            d = np.linalg.norm(cent - v, axis=1)
            order = np.argsort(d)
            best = int(order[0])
            # NOT cosine: the stored centroids are means of unit vectors and
            # were never renormalised, so their norms run 0.79-0.96. cent @ v
            # is therefore cosine scaled by the centroid's norm. That is fine
            # for ranking the six against each other, and it is why the floor
            # below is calibrated on these same numbers rather than on a 0-1
            # cosine scale.
            sims = cent @ v

            # The nearest of six centroids is always *some* category, even for
            # a product this model has never seen - "Nike running shoes" used
            # to come back as Chargers & cables. So before naming a category,
            # check that the input resembles something the clustering was
            # actually fitted on. Two tests, and an input must pass both:
            #
            #   1. distance to the nearest centroid  (is it near a cluster?)
            #   2. distance to the nearest REAL item (is it near real data?)
            #
            # Neither alone is enough, and which one is weaker depends on the
            # model. Measured on MiniLM-L12 with 8 in-range phrasings and 11
            # off-range ones (phones, laptops, shoes, headphones, coffee pods,
            # a banana):
            #
            #   product names    test 1  in-range 0.372-0.656, off-range <=0.349
            #                    test 2  in-range 0.572-0.817, off-range <=0.590
            #   category strings test 1  in-range 0.502-0.807, off-range <=0.620
            #                    test 2  in-range 0.732-1.000, off-range <=0.668
            #
            # Test 1 separates product names but overlaps on category strings;
            # test 2 is the other way round. Together they separate both. The
            # product-name margin is thin - 0.372 against 0.349 - so this is a
            # calibrated guard, not a guarantee.
            #
            # Test 2's single false accept is instructive: "Logitech wireless
            # keyboard" scores 0.590 because the dataset contains "Kindle
            # Keyboard". Test 1 rejects it. The app prints whichever real item
            # was matched, so a borderline call is visible rather than hidden.
            CENT_FLOOR, NN_FLOOR = (0.45, 0.70) if USING_CAT else (0.36, 0.50)
            memb = load_members(USING_CAT)
            nn = memb @ v
            j = int(np.argmax(nn))
            if USING_CAT:
                nn_label = tbl["category_string"].iloc[j]
                nn_cat = tbl["meta_category"].iloc[j]
                THING = "category string"
            else:
                nn_label = prods["product_name"].iloc[j]
                nn_cat = prods["category"].iloc[j]
                THING = "product"
            known = sims[best] >= CENT_FLOOR and nn[j] >= NN_FLOOR

            st.markdown("### Category")
            if known:
                st.markdown(
                    f"<span style='font-size:2.2rem;font-weight:700;color:{FLUX_DK}'>"
                    f"{cnames[best]}</span>", unsafe_allow_html=True)
                st.caption(
                    f"Nearest of the six centroids, score {sims[best]:.3f}. "
                    f"Runner-up: **{cnames[int(order[1])]}** at {sims[int(order[1])]:.3f}."
                )
                st.success(
                    f"Closest real {THING} in the dataset: **{str(nn_label)[:70]}** "
                    f"at {nn[j]:.3f} cosine, and it sits in *{nn_cat}*."
                )
                if nn_cat != cnames[best]:
                    st.warning(
                        f"The nearest centroid says **{cnames[best]}** but the "
                        f"nearest real {THING} sits in **{nn_cat}**. The two "
                        f"disagree, so treat this one as a boundary case."
                    )
            else:
                st.markdown(
                    "<span style='font-size:2.2rem;font-weight:700;color:#8a8a8a'>"
                    "No category matches</span>", unsafe_allow_html=True)
                failed = []
                if sims[best] < CENT_FLOOR:
                    failed.append(
                        f"it is {sims[best]:.3f} from the nearest centroid "
                        f"(*{cnames[best]}*), under the {CENT_FLOOR:.2f} floor")
                if nn[j] < NN_FLOOR:
                    failed.append(
                        f"the closest real {THING} is **{str(nn_label)[:60]}** "
                        f"at only {nn[j]:.3f}, under the {NN_FLOOR:.2f} floor")
                st.error(
                    "Nothing in the dataset is close enough to this: "
                    + ", and ".join(failed) + ". The clustering was fitted on "
                    + ("38 Amazon category strings" if USING_CAT
                       else "39 Amazon devices")
                    + ", so this sits outside it. Naming one of the six would "
                      "be inventing an answer rather than reporting one."
                )
                st.caption(
                    f"Before this check the app would have answered "
                    f"*{cnames[best]}* here."
                )

            if known:
                gap = sims[best] - sims[int(order[1])]
                if gap < 0.05:
                    st.warning(
                        f"The top two centroids are only {gap:.3f} apart, so "
                        f"this one sits near a boundary between categories."
                    )

                st.markdown("### What is already in that category")
                if USING_CAT:
                    members = (tbl[tbl["meta_category"] == cnames[best]]
                               .sort_values("reviews", ascending=False))
                    st.dataframe(members[["category_string", "reviews"]],
                                 hide_index=True, use_container_width=True)
                else:
                    members = (prods[prods["category"] == cnames[best]]
                               .sort_values("reviews", ascending=False))
                    st.dataframe(
                        members[["product_name", "reviews", "mean_rating"]]
                        .style.format({"mean_rating": "{:.2f}"}),
                        hide_index=True, use_container_width=True)

            st.markdown("### Similarity to every category")
            st.dataframe(
                pd.DataFrame({"category": [cnames[i] for i in order],
                              "similarity": [sims[i] for i in order]})
                .style.format({"similarity": "{:.3f}"}),
                hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### The six categories")
    if USING_CAT:
        summary = (tbl.groupby("meta_category")
                   .agg(category_strings=("category_string", "size"),
                        reviews=("reviews", "sum"))
                   .sort_values("reviews", ascending=False).reset_index())
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption(
            "All **34,626** reviews are covered, against 27,866 for the "
            "product-name model. The extra 6,760 are reviews whose product "
            "name is empty — they land in *TV & home theatre*, which is what "
            "their category string describes."
        )
    else:
        summary = (prods.groupby("category")
                   .agg(products=("product_name", "size"), reviews=("reviews", "sum"),
                        mean_rating=("mean_rating", "mean"))
                   .sort_values("reviews", ascending=False).reset_index())
        st.dataframe(summary.style.format({"mean_rating": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
        st.caption(
        "Agreement with a hand-labelled answer, measured by adjusted Rand "
        "index: **0.8659**. TF-IDF over the same names scored 0.5087, and the "
        "first attempt — product name plus Amazon's own category string — "
        "scored 0.0446, which is no better than labelling at random."
    )

# ================================================================= tab 3
with TAB[2]:
    st.subheader("What are customers saying about each category?")
    st.write(
        "Task 3. One article per meta-category from Task 2. Everything "
        "countable is **computed from the data** - review counts, ratings, top "
        "products, the product people complain about most, the complaint "
        "themes. A generative model is called **once**, at the end, for the "
        "single sentence that needs judgement, and it is labelled where it "
        "appears."
    )

    SUMM = load_summaries()
    order_by_size = sorted(SUMM, key=lambda c: -SUMM[c]["reviews"])
    cat = st.selectbox("Category", order_by_size, key="summary_cat")
    s = SUMM[cat]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", format(s["reviews"], ","))
    c2.metric("Mean rating", "%.2f" % s["rating"])
    c3.metric("Positive", "%.1f%%" % s["pct_positive"])
    c4.metric("Negative", "%.1f%%" % s["pct_negative"])
    st.caption("%d products in this category." % s["n_products"])

    # ---------------------------------------------------------- top 3
    st.markdown("### Most reviewed products")
    st.dataframe(
        pd.DataFrame(s["top3"]).rename(columns={
            "name": "product", "reviews": "reviews",
            "rating": "mean rating", "neg_rate": "% negative"})
        .style.format({"mean rating": "{:.2f}", "% negative": "{:.1f}%"}),
        hide_index=True, use_container_width=True)

    # ---------------------------------------------------------- worst
    st.markdown("### The one to avoid")
    w = s["worst"]
    if w is None:
        st.info(
            "No product in this category has %d or more reviews, so there is "
            "not enough evidence to single one out." % 20)
    elif w["stands_out"]:
        st.error(
            "**%s** - %.1f%% of its %s reviews are negative, against %.1f%% "
            "for the category, at %.2f stars."
            % (w["name"], w["neg_rate"], format(w["reviews"], ","),
               s["pct_negative"], w["rating"]))
    else:
        st.success(
            "Nothing stands out. The highest negative rate in this category is "
            "%.1f%% (%s), against %.1f%% for the category as a whole - not far "
            "enough above average to call it the worst."
            % (w["neg_rate"], w["name"], s["pct_negative"]))
        st.caption(
            "Ranking by negative rate, not by mean star. Every product here "
            "rates 4.4-4.8, so taking the minimum rating just returns whichever "
            "best-seller has the most reviews - which is what the first version "
            "of this notebook did."
        )

    # ---------------------------------------------------------- complaints
    st.markdown("### What the complaints are about")
    if not s["complaints"]:
        st.info(
            "No negative reviews in this category at all, so there is nothing "
            "to mine." if s["n_negative"] == 0 else
            "Only %d negative reviews here - under the 25 needed before a "
            "log-odds comparison means anything." % s["n_negative"])
    else:
        cdf = pd.DataFrame(s["complaints"]).rename(columns={
            "word": "term", "log_odds": "how much more likely in a complaint",
            "n_reviews": "complaints containing it"})
        st.dataframe(
            cdf, hide_index=True, use_container_width=True,
            column_config={
                "how much more likely in a complaint": st.column_config.ProgressColumn(
                    "log-odds vs praise", format="%.2f",
                    min_value=0.0, max_value=float(max(
                        c["log_odds"] for c in s["complaints"])) * 1.05)})
        st.caption(
            "Log-odds ratio with Dirichlet smoothing: how much more likely a "
            "word is in this category's %d negative reviews than in its "
            "positive ones. Counted once per review, and a term must appear in "
            "at least %d of them - otherwise two chatty reviews invent a theme."
            % (s["n_negative"], s["min_df"]))

    # ---------------------------------------------------------- quotes
    st.markdown("### What unhappy customers actually said")
    if not s["quotes"]:
        st.info("No negative reviews in this category at all.")
    else:
        if s["evidence_rule"] == "most helpful":
            st.caption(
                "The three complaints other shoppers voted most helpful - not "
                "a random sample. `reviews.numHelpful` is populated for 98.6% "
                "of rows.")
        else:
            st.caption(
                "Nobody voted on the complaints here, so these are the "
                "lowest-rated ones instead.")
        for q in s["quotes"]:
            votes = ("%d readers found this helpful" % q["helpful"]
                     if q["helpful"] else "%.0f stars" % q["rating"])
            head = ("**%s** - " % q["title"]) if q["title"] else ""
            st.markdown("> %s%s\n>\n> *%s*" % (head, q["text"][:700], votes))

    # ---------------------------------------------------------- the model
    st.markdown("### The generated line")
    if s["generated"]:
        st.markdown(
            "<div style='border-left:4px solid %s;padding:.6rem .9rem;"
            "background:rgba(57,255,20,.07)'>%s</div>"
            % (FLUX_DK, s["generated"]), unsafe_allow_html=True)
        st.caption(
            "This sentence, and only this sentence, was written by `%s`. It was "
            "given the three quotes above and nothing else - no product names, "
            "no counts - because those are already known exactly and a model "
            "asked for them will guess." % s["model"])
    else:
        st.info(
            "Nothing generated for this category: there are too few words of "
            "complaint to summarise. The article above is entirely computed.")

    # ---------------------------------------------------------- honesty
    with st.expander("Where this still goes wrong"):
        st.markdown(
            """
**The quotes can belong to the wrong product.** The most-helpful negative
review filed under *Echo (White)* is actually about a refurbished Fire TV, and
9 of the 40 negative reviews under *Streaming* talk about an Echo. The product
and the review text do not always match in the source data. Because the quotes
are ranked by helpfulness, one mismatched row with 292 votes can set the tone
for a whole category - which is exactly what happens to **Smart speakers**.

**distilbart is more extractive than it looks.** Asked to compress three
complaints it often returns their sharpest sentences stitched together rather
than genuinely new prose. That is honest output, but it is closer to selection
than to writing.

**Cases & covers has 23 reviews and no negatives at all**, so there is nothing
to mine and nothing to generate. The tab says so instead of inventing a theme.

**Complaint terms are words, not reasons.** *returned*, *waste*, *horrible*
tell you a complaint happened, not what broke. Reading the quotes is still
necessary.
            """)

    with st.expander("The one-paragraph version (what the dashboard consumes)"):
        st.caption(
            "`category_summaries.json` holds these as flat `{category: text}` "
            "strings, the same shape Rayhan's dashboard already reads, so it "
            "picks them up with no code change.")
        st.code(build_flat(cat, s), language=None)

# ================================================================= tab 4
with TAB[3]:
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
