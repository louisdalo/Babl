"""
    Recommendation page (shown through app.py).
    - Needs TARGET labels before recommending anything.
    - Preprocesses the books once, re-trains the model whenever labels.csv
      (or the books files) change, saves model + metrics in model/,
      then shows the 10 best matches and the model analysis.
"""

##### IMPORTS ##### ------------------------------------------------------------
import sys
import altair as alt
import pandas as pd
import streamlit as st

from utils.paths import (BOOKS_PATH, HANDCRAFTED_BOOKS_PATH, LABELS_PATH, 
                         MODEL_DIR)
from utils.data import (
    TARGET, append_label, ensure_handcrafted_file, file_signature, load_books, 
    read_labels)
from utils.cover import cover
from utils.theme import theme_color

TOP_N = 10
CAT_PALETTE = ["#606c38","#283618","#fefae0","#dda15e","#bc6c25"]

# Streamlit session parameters
ss = st.session_state
ss.setdefault("skipped_recs", set()) # recommendations hidden for this session


##### FUNCTIONS ##### ----------------------------------------------------------
### Cached functions -----------------------------------------------------------
@st.cache_resource
def _memory():
    """Creates a shared dict that persist accross streamlit runs (not sessions)."""
    return {}

def memo(kind: str, key, compute):
    """
        Stores results of a "heavy" computing steps in _memory().
        If "kind" of data exists and its key match, no need to recompute.

        Parameters
        ----------
        kind: str
            The label of the object being stored.
        key: a tuple
            The output of utils.data.file_signature().
        compute: a function
            The function to execute if "kind" object with given key does not 
            already exist in _memory()
        
        Updates 
        -------
        _memory().

        Returns
        -------
        The result of compute() (or its cached result).
    """
    store = _memory()
    if kind not in store or store[kind][0] != key:
        store[kind] = (key, compute())
    return store[kind][1]

class Loader:
    """
        Displays a progress bar with a status line under it when something 
        heavy is being computed.
    """

    def __init__(self):
        """
            Initialisation of progress bar and status messages.
            Sets features_computed to False (computation has not run already). 
        """
        self._bar, self._status = st.empty(), st.empty()
        self.features_computed = False

    def update(self, fraction: float, message: str):
        """
            Updates the progress bar and display the current status.

            Parameters
            ----------
            fraction: float
                The overall progress as a fraction from 0.0 to 1.0.
                Values outside this range are clamped.
            message: str
                A description of the task currently being performed.
        """
        fraction = min(max(fraction, 0.0), 1.0)
        self._bar.progress(fraction)
        self._status.caption(f"{message} ({fraction:.0%})")

    def phase(self, start, end):
        """
            Creates a progress callback for a specific stage of the workflow.

            Maps a subtask's progress, from 0.0 to 1.0, onto the specified
            interval of the overall progress bar.

            Parameters
            ----------
            start: numeric
                Starting point of the stage on the overall progress bar.
            end: numeric
                Ending point of the stage on the overall progress bar.
            
            Returns
            -------
                A callback accepting a progress fraction and status message.
                Calling it updates the overall progress bar accordingly.
        """
        return lambda fraction, message: self.update(start + (end - start) * fraction, message)

    def clear(self):
        """
            Removes the progress bar and status message from the Streamlit UI.

            Clears both placeholders so the progress indicators are no longer
            displayed after computation has finished or been interrupted.
        """
        self._bar.empty()
        self._status.empty()


### Heavy steps ----------------------------------------------------------------
# WARNING: some functions [build_features(), train_model(), save_result(), 
# NotEnoughData] in the following functions are imported later.
def get_features(books_sig, loader):
    """
        Loads books and builds features, reusing cached results when possible.

        Uses books_sig to determine whether previously computed features can
        be reused. If the signature changes or no cached result exists, loads
        the books and builds their features while reporting progress.

        Parameters
        ----------
        books_sig: a tuple.
            Usually, should be the output of utils.data.file_signature(). Its A 
            signature identifying the current book data, used to validate the cached 
            features.
        loader: Loader instance
            A Loader instance used to report computation progress and track whether 
            features were already computed during this run.

        Returns
        -------
        The features used by the model.
    """
    def compute():
        """
            Loads books and compute their features with progress updates.

            Marks feature computation when started, updates the progress display,
            loads the book data, and builds features while mapping progress to
            the 5%-60% range of the overall progress bar.

            Returns
            -------
            The features for the selected books.
        """
        loader.features_computed = True
        loader.update(0.02, "Loading books...")
        books = load_books()
        return build_features(books, progress=loader.phase(0.05, 0.60))
    return memo("features", books_sig, compute)

def get_model(books_sig, labels_sig, feats, loader):
    """
        Retrieves or train the model and saves its results when successful.

        Uses the book and label signatures to determine whether a cached
        model result can be reused.

        Parameters
        ----------
        books_sig: a tuple
            A signature identifying the current book data.
            (See utils.data.file_signature()).
        labels_sig: a tuple
            A signature identifying the current label data. 
            (See utils.data.file_signature()).
        feats: matrix
            The features used to train the model. (See get_features()).
        loader: Loader instance
            Used to report progress and determine the starting point of the 
            training progress bar.

        Returns
        -------
        The trained model result, or a NotEnoughData exception object
        if there is insufficient data to train the model.
    """
    def compute():
        """
            Do exactly what its parent function says if trained model is not
            available in cache.
        """
        start = 0.60 if loader.features_computed else 0.0
        loader.update(start, "Loading labels...")
        rows = read_labels()
        split = start + (1.0 - start) * 0.9
        try:
            result = train_model(feats, rows, progress=loader.phase(start, split))
        except NotEnoughData as exc:
            return exc
        loader.update(split, "Saving model and metrics...")
        save_result(feats, result, rows, MODEL_DIR)     # model.joblib, metrics.json, ...
        return result
    return memo("model", (books_sig, labels_sig), compute)


### ACTIONS ##### --------------------------------------------------------------
def rate_recommended(book, label):
    """
        When the user already read a recommended book: store the label 
        (the model retrains on rerun).

        Parameters
        ----------
        book:
            The key corresponding to a book from load_books().
        label:
            The label for this book (see ACTIONS in rating.py).

        Updates
        -------
        LABELS_PATH
        st.toast
    """
    append_label(book, label, 1)
    verdict = {1.0: "liked", 0.5: "neutral", 0.0: "disliked"}[label]
    st.toast(f"Saved: ‘{book['title']}’ ({verdict}). The model is being updated.")

def interested(book):
    """
        When the user did not read a recommended book but likes the
        recommendation: store the book as "Interested" (like "Interested" 
        on the Annotation page).
        (the model retrains on rerun).

        Parameters
        ----------
        book:
            The key corresponding to a book from load_books().

        Updates
        -------
        LABELS_PATH
        st.toast
    """
    append_label(book, 1.0, 0)
    st.toast(f"Saved: ‘{book['title']}’ (interested). The model is being updated.")

def not_interested(book):
    """
        When the user did not read a recommended book but does not like the
        recommendation: store the book as "not interested"(like "Not for me" 
        on the Annotation page).
        (the model retrains on rerun).

        Parameters
        ----------
        book:
            The key corresponding to a book from load_books().

        Updates
        -------
        LABELS_PATH
        st.toast
    """
    append_label(book, 0.0, 0)
    st.toast(f"Saved: ‘{book['title']}’ (not interested). The model is being updated.")

def skip_recommended(book_id):
    """
        Hide this recommendation for the session, without telling the model 
        anything (no change on labels).

        Parameters
        ----------
        book_id:
            The key corresponding to a book from load_books().

        Updates
        -------
        skipped_recs
    """
    ss.skipped_recs.add(book_id)


##### PLOTS ##### --------------------------------------------------------------
def plot_history(hist):
    """
        Creates a line chart of cross-validated performance history.

        Plots performance against the number of available labels. 
        Displays points at recorded observations.

        Parameters
        ----------
        hist: pd.DataFrame
            Muist contain at least 4 columns: 'cv_balanced_accuracy', 
            'n_labels', 'trained_at' and 'auc'.

        Returns
        -------
        An Altair chart containing the performance history line 
        (can be displayed directly by Streamlit).
    """
    df = hist.dropna(subset=["cv_balanced_accuracy", "cv_std"]).copy()

    df["lower"] = (df["cv_balanced_accuracy"] - df["cv_std"]).clip(lower=0)
    df["upper"] = (df["cv_balanced_accuracy"] + df["cv_std"]).clip(upper=1)

    y_min = max(0, df["lower"].min() - 0.025)
    y_max = min(1, df["upper"].max() + 0.025)

    x = alt.X(
        "n_labels:Q",
        title="Number of labels",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(
            tickMinStep=1,
            format="d")
    )
    y = alt.Y(
        "cv_balanced_accuracy:Q",
        title="Balanced accuracy",
        scale=alt.Scale(domain=[y_min, y_max]),
        axis=alt.Axis(format="%"),
    )

    ribbon = alt.Chart(df).mark_area(
        opacity=0.2,
        color=theme_color("primaryColor"),
    ).encode(
        x=x,
        y=alt.Y("lower:Q", title="Balanced accuracy",
                scale=alt.Scale(domain=[y_min, y_max]),
                axis=alt.Axis(format="%")),
        y2="upper:Q",
        tooltip=[
            alt.Tooltip("upper:Q", format=".1%", title="Mean + 1 SD"),
            alt.Tooltip("lower:Q", format=".1%", title="Mean - 1 SD"),
            "n_labels",
        ],
    )

    line = alt.Chart(df).mark_line(
        color=theme_color("primaryColor")
    ).encode(
        x=x,
        y=y,
        tooltip=[
            alt.Tooltip("cv_balanced_accuracy:Q", format=".1%"),
            alt.Tooltip("cv_std:Q", format=".1%", title="Standard deviation"),
            alt.Tooltip("auc:Q", format=".2f"),
            "n_labels",
        ],
    )

    points = alt.Chart(df).mark_point(
        color=theme_color("primaryColor"),
        filled=True
    ).encode(
        x=x,
        y=y,
        tooltip=[
            alt.Tooltip("cv_balanced_accuracy:Q", format=".1%"),
            alt.Tooltip("cv_std:Q", format=".1%", title="Standard deviation"),
            alt.Tooltip("auc:Q", format=".2f", title="AUC"),
            "n_labels",
        ],
    )

    return (ribbon + line + points).properties(
        width=800,
        height=400
    )

def plot_folds(scores, mean):
    """
        Creates an Altair boxplot showing balanced accuracy from 
        cross-validations. A dashed red line shows the baseline score (random 
        guesses) and a solid black line shows the mean score accross splits.


        Parameters
        ----------
        scores: set/list
            A sequence of balanced accuracy scores, one per split.
        mean: numeric
            Mean balanced accuracy across the cross-validation splits.

        Returns
        -------
        An Altair layered chart object containing the boxplot and reference
        lines for the baseline and mean performance (can be displayed directly 
        by Streamlit).
    """
    s = pd.Series(scores, dtype=float)

    # Same statistics as Vega-Lite's boxplot: quartiles with linear
    # interpolation, whiskers reach the furthest points within 1.5 * IQR,
    # anything beyond is an outlier.
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    inside = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
    lo, hi = inside.min(), inside.max()

    stats = pd.DataFrame({
        "CV": [""], "lo": [lo], "q1": [q1], "median": [med], "q3": [q3], "hi": [hi]
    })
    out = s[(s < lo) | (s > hi)]
    outliers = pd.DataFrame({
        "CV": [""] * len(out),
        "balanced accuracy": out.values
    })

    color = theme_color("primaryColor")
    x_scale = alt.Scale(domain=[s.min() - 0.025, s.max() + 0.025])
    y = alt.Y("CV:N", title=None)
    fmt = ".1%"

    tip = [
        alt.Tooltip("lo:Q", format=fmt, title="Lower whisker"),
        alt.Tooltip("q1:Q", format=fmt, title="Q1"),
        alt.Tooltip("median:Q", format=fmt, title="Median"),
        alt.Tooltip("q3:Q", format=fmt, title="Q3"),
        alt.Tooltip("hi:Q", format=fmt, title="Upper whisker"),
    ]

    # Whiskers (also defines the x axis shared by all layers)
    whisker = alt.Chart(stats).mark_rule(color=color).encode(
        x=alt.X("lo:Q", title="Balanced accuracy", scale=x_scale,
                axis=alt.Axis(format="%")),
        x2="hi:Q",
        y=y,
        tooltip=tip,
    )

    # Box: Q1 -> Q3
    box = alt.Chart(stats).mark_bar(color=color, size=50, opacity=1).encode(
        x="q1:Q", x2="q3:Q", y=y, tooltip=tip,
    )

    # Median line
    median = alt.Chart(stats).mark_tick(
        color="white", thickness=2, size=50
    ).encode(
        x="median:Q", y=y, tooltip=tip,
    )

    # Outliers (only the raw value is shown, formatted)
    points = alt.Chart(outliers).mark_point(color=color, size=40).encode(
        x=alt.X("balanced accuracy:Q", scale=x_scale),
        y=y,
        tooltip=[alt.Tooltip("balanced accuracy:Q", format=fmt, title="Outlier")],
    )

    base = alt.Chart(
        pd.DataFrame({"x": [BASELINE]})
    ).mark_rule(
        color="red",
        opacity=1,
        strokeDash=[5, 4]
    ).encode(x="x:Q")

    avg = alt.Chart(
        pd.DataFrame({"x": [mean], "CV": [""]})
    ).mark_point(
        color="black",
        shape="diamond",
        opacity=1,
        size=150,
        filled=True
    ).encode(
        x="x:Q",
        y=alt.Y("CV:N"),
        tooltip=[alt.Tooltip("x:Q", format=fmt, title="Mean")],
    )

    return (whisker + box + median + points + base + avg).properties(
        width=400,
        height=150
    )

def plot_roc(roc, auc):
    """
        Creates a ROC curve.

        Aka: plots the false positive rate against the true positive rate at 
        different classification thresholds. Adds a dashed diagonal baseline 
        representing a random-classifier performance.


        Parameters
        ----------
        scores: set/list
            A sequence of balanced accuracy scores, one per split.
        mean: numeric
            Mean balanced accuracy across the cross-validation splits.

        Returns
        -------
        An Altair layered chart object containing the score bars and reference
        lines for the baseline and mean performance (can be displayed directly 
        by Streamlit).
    """
    df = pd.DataFrame(roc)
    line = alt.Chart(df).mark_line(
        interpolate="step-after",
        color=theme_color("primaryColor")
    ).encode(
        x=alt.X(
            "fpr:Q", 
            title="False positive rate", 
            scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(
            "tpr:Q", 
            title="True positive rate", 
            scale=alt.Scale(domain=[0, 1]))
    )
    diag = alt.Chart(
        pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    ).mark_line(
        color="red", 
        strokeDash=[5, 4]
    ).encode(
        x="x:Q", 
        y="y:Q"
        )
    return (line + diag).properties(width=400, height=400)

def plot_coef(df, title):
    """
        Shows each feature's coefficient, with bars colored by block. The bar 
        order depends on the sign of the first coefficient in df.
    
        Parameters
        ----------
        df: pd.dataframe
            Should contain at least 5 columns: name, feature, influence, coef, 
            and block.
        title: str
            The title of the graph.

        Returns
        -------
        an alt.Chart object (can be displayed directly by Streamlit).
    """
    return alt.Chart(
            df, 
            title=alt.Title(title, offset=0, anchor="middle")
        ).mark_bar(
        ).encode(
            x=alt.X(
                "coef:Q", 
                title="Coefficient"),
            y=alt.Y(
                "name:N", 
                sort=alt.EncodingSortField(
                    "coef", 
                    order="descending" if df["coef"].iloc[0] > 0 else "ascending"
                    ), 
                title=None,
                axis=alt.Axis(labelLimit=500)),
            color=alt.Color(
                "block:N", 
                title=None,
                scale=alt.Scale(
                    domain=["subjects", "summary words", "authors", "numeric"],
                    range=CAT_PALETTE
                    ),
                legend=alt.Legend(
                    direction="horizontal",
                    orient="top",
                    padding=0)
                ),
            tooltip=["feature", alt.Tooltip("coef:Q", format=".3f"), alt.Tooltip("influence:Q", format=".4f")],
        )

def plot_influence(df):
    """
        Creates an Altair horizontal bar chart showing the influence of 
        individual features on model predictions, grouped visually by their 
        block (see plot_blocks).

        Parameters
        ----------
        df: pd.DataFrame
            Should contain at least 5 columns: name, feature, influence, coef, 
            and block.

        Returns
        -------
        an alt.Chart object (can be displayed directly by Streamlit).
    """
    return alt.Chart(
        df,
        title=alt.Title(
            "Most influential features overall", 
            anchor="middle")
    ).mark_bar().encode(
        x=alt.X("influence:Q", title="Mean |coefficient × value| over all books"),
        y=alt.Y("name:N", sort="-x", title=None),
        color=alt.Color(
            "block:N", 
            title=None,
            scale=alt.Scale(
                domain=["subjects", "summary words", "authors", "numeric"],
                range=CAT_PALETTE),
            legend=alt.Legend(
                direction="horizontal",
                orient="bottom",
                padding=0)
            ),
        tooltip=["feature", alt.Tooltip("influence:Q", format=".4f"), alt.Tooltip("coef:Q", format=".3f")],
    )

def plot_confusion(cm):
    """
        Creates a confusion-matrix heatmap for out-of-fold predictions.


        Parameters
        ----------
        cm: matrix
            A 2x2 matrix ordered as: disliked, liked for both axes.

        Returns
        -------
        An Altair layered chart object containing the heatmap and cell counts
        (can be displayed directly by Streamlit).
    """
    df = pd.DataFrame([
        {"actual": a, "predicted": p, "books": cm[i][j]}
        for i, a in enumerate(["disliked", "liked"])
        for j, p in enumerate(["disliked", "liked"])
    ])

    base = alt.Chart(df).encode(
        x=alt.X(
            "predicted:N",
            sort=["disliked", "liked"],
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "actual:N",
            sort=["disliked", "liked"],
            axis=alt.Axis(labelAngle=-90)
        )
    )

    heatmap = base.mark_rect(
        opacity=0.65
    ).encode(
        color=alt.Color(
            "books:Q",
            legend=None,
            scale=alt.Scale(range=[theme_color("backgroundColor"), theme_color("primaryColor")])
        )
    )

    text = base.mark_text(fontSize=20).encode(
        text="books:Q"
    )

    return (heatmap + text).properties(
        width=400,
        height=400
    )

def plot_oof(oof):
    """
        Creates a histogram of out-of-fold predicted probabilities.

        Groups predicted probabilities into bins and displays the number
        of books in each bin. Uses different colors to distinguish books
        actually labeled as disliked or liked.

        Parameters
        ----------
        oof: DataFrame 
            Must contains at least 2 columns: 'proba' and 'actual'.

        Returns
        -------
        An Altair histogram showing predicted probability distributions for the 
        (two) actual classes (can be displayed directly by Streamlit).
    """
    graph = alt.Chart(oof).mark_bar(
        opacity=0.65
    ).encode(
        x=alt.X(
            "proba:Q", 
            bin=alt.Bin(maxbins=10), 
            title="Predicted probability of liking"),
        y=alt.Y(
            "count()", 
            stack=None, 
            title="Books"),
        color=alt.Color(
            "actual:N", 
            title="You said", 
            scale=alt.Scale(
                domain=["disliked", "liked"], 
                range=["#e45756", "#54a24b"])))

    return graph.properties(
        width=400,
        height=400
    )


##### UI ##### -----------------------------------------------------------------
def render_analysis(result):
    """
        Renders an area with 4 tabs, dedicated to analysing a model's metrics.

        Parameters
        ----------
        results:
            The output of get_model(), a Result object (see recommender.py).
    """
    # Extract importance from results
    imp_all = result.importance
    imp = imp_all[imp_all["block"] != "sentence embeddings"]     # 384 anonymous dimensions: not readable
    
    # Initialise tabs
    tab_hist, tab_perf, tab_feat, tab_errors = st.tabs([
        "Training history", 
        "Training performance", 
        "Importance of variables",
        "Confusions"])

    # Tab with balanced accuracy history
    with tab_hist:
        hist = read_history(MODEL_DIR)
        if hist["cv_balanced_accuracy"].notna().sum() < 2:
            st.info("The curve appears once the model has been trained on at least two different sets of labels.")
        else:
            # centering hack
            _, col2, _ = st.columns([1, 6, 1])
            col2.markdown("**Training history of model**")
            col2.altair_chart(plot_history(hist), width="stretch", theme="streamlit")
            col2.caption("This graph shows how the balanced accuracy of the model "
                       "evolves with the number of books you annotate. "
                       "*Logically* the more books you annotate, the more the "
                       "model can successfully generalise your tastes, and "
                       "therefore improve.\n\n")

    # Tab with detailed scores of the current model
    with tab_perf:
        if result.cv_scores is None:
            st.info("Cross-validation needs at least 2 liked and 2 disliked books.")
        else:

            # centering hack
            _, col2, _, col4, _ = st.columns([1, 3, 1, 3, 1])
            with st.container():
                st.markdown("""
                    <style>
                    div[data-testid="stHorizontalBlock"]:has(
                        div[data-testid="stColumn"]
                    ) {
                        align-items: center;
                    }
                    </style>
                """, unsafe_allow_html=True)

                col2.markdown(f"**Balanced accuracies ({CV_FOLDS * CV_REPEATS}"
                              "-fold cross-validation)**")
                col2.altair_chart(
                    plot_folds(result.cv_scores, result.cv_bal_acc),
                    width="stretch",
                    theme=None
                )
                col4.markdown("**Receiver Operating Characteristic (ROC) curve**")
                col4.altair_chart(
                    plot_roc(result.roc, result.auc),
                    width="stretch",
                    theme=None
                )

            _, col2, _, col4, _ = st.columns([1, 3, 1, 3, 1])
            col2.caption(
                "Boxplot showing the distribution of the balanced accuracies "
                f"computed for {CV_FOLDS * CV_REPEATS} cross-validations*. "
                "The closer to 100%, the better.")
            col4.caption("The ROC curve is the plot of the true positive rate "
                         "against the false positive rate at each threshold "
                         f"setting. Its total ({result.auc:.2f}) is the Area "
                         "Under the Curve (AUC). The closer to 1 (whole area), "
                         "the better.")
            st.caption("*The red dashed line shows what the score would be for random " 
                       "guesses.*")
            st.caption(
                "*Cross-validating is a common method in Machine Learning used to "
                "evaluate a model's performance by partitioning its full training "
                "dataset into smaller subsets for training and tesing. This "
                "shows how good the model can generalise from its training data.")
            
    # Tab with the importance of each feature for predictions
    with tab_feat:
        # st.markdown("**Weight of each family of variables**")
        st.text("These graphs show how much a book's feature influences the "
                "model's prediction towards what you might like or dislike.",
                help="IMO the most interesting tab of all, it tells you "
                     "***why*** the model thinks you like and dislike a book!")
        st.caption("Legend: coefficient = direction and strength of a feature. "
                   "Influence = how much it actually moves predictions "
                    "across the catalogue.")
        st.text("")
        st.text("")        
        _, c1, c2, _ = st.columns([1, 3, 3, 1])
        block = c1.selectbox("Variables to show", ["All", *sorted(imp["block"].unique())], 0)
        n = c2.slider("Features per chart", 5, 30, 10)
        sub = imp if block == "All" else imp[imp["block"] == block]
        left, right = st.columns(2)
        pos, neg = sub.nlargest(n, "coef"), sub.nsmallest(n, "coef")
        left.altair_chart(plot_coef(pos, "Features that push predictions towards LIKE"), width="stretch")
        right.altair_chart(plot_coef(neg, "Features that push predictions towards DISLIKE"), width="stretch")
        st.altair_chart(plot_influence(sub.nlargest(n, "influence")), width="stretch")
        st.caption("Sentence embeddings are counted in the first chart but not listed below: their dimensions "
                   "have no readable meaning (see \"Close to a book you liked\" on each recommendation instead).")
        

    with tab_errors:
        # centering hack
        _, col2, _, col4, _ = st.columns([1, 3, 0.5, 4, 1])
        with st.container():
            st.markdown("""
                <style>
                div[data-testid="stHorizontalBlock"]:has(
                    div[data-testid="stColumn"]
                ) {
                    align-items: center;
                }
                </style>
            """, unsafe_allow_html=True)
        col2.markdown(f"**Confusions matrix**")
        col2.altair_chart(plot_confusion(result.confusion), width="stretch", theme=None)
        col2.caption(
            "Each cell shows the number of labels the model attributed to a "
            f"category. *For example, we can read that {result.confusion[1][1]} "
            "of the books you said you actually liked are predicted as books "
            "that you **should** like according to the model, while it expected "
            f"that {result.confusion[0][1]} of the books you liked would not "
            "fit your tastes.*")
        
        col4.markdown(f"**Predicted P(like) of the books you rated (out-of-fold)**")
        col4.altair_chart(plot_oof(result.oof), width="stretch", theme=None)
        col4.caption(
            "This plot shows how the model actually classifies the books you "
            "rated.\n\nThe model computes a probability for each book, based on "
            "your annotations. The more your liked/disliked distributions on "
            "this plot are bi-modal (aka seperated), the better will be the "
            "predictions.")


##### PAGE ##### ---------------------------------------------------------------
### Initialisation of page -----------------------------------------------------
ensure_handcrafted_file()
st.title("Recommendations")

# Fetch labels, if not enough in memory, go no further.
labels = read_labels()
n = len(labels)
if n < TARGET:
    st.info("Rate more books to receive personalised recommendations !", icon=":material/info:")
    st.progress(min(n / TARGET, 1.0), text=f"{n} / {TARGET} labels")
    st.page_link("rating.py", label="Go rate some books", icon=":material/edit_note:")
    st.stop()

# Load sk-learn packages, utils.recommender, features and model.
# Heavy computation involved, so show a progress bar.
loader = Loader()
try:
    if "utils.recommender" not in sys.modules:      
        # first visit: importing scikit-learn takes a few seconds, show progress
        loader.update(0.0, "Loading packages...")
    from utils.recommender import (BASELINE, CV_FOLDS, CV_REPEATS, 
                                   NotEnoughData, build_features, read_history, 
                                   recommend, save_result, train_model)

    books_sig = (file_signature(BOOKS_PATH), file_signature(HANDCRAFTED_BOOKS_PATH))
    labels_sig = file_signature(LABELS_PATH)
    feats = get_features(books_sig, loader)
    result = get_model(books_sig, labels_sig, feats, loader)
finally:
    loader.clear()

# Potential errors to show
if feats.embedding_error:
    st.warning(f"Sentence embeddings are not used: {feats.embedding_error} "
               "Recommendations rely on TF-IDF features only.", icon=":material/warning:")

if isinstance(result, NotEnoughData):
    st.warning(f"{result} Rate a few books of the other kind to unlock recommendations.",
               icon=":material/warning:")
    st.stop()


### Side bar -------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    max_top = st.slider("Number of recommendations to show", 0, 100, TOP_N, step=1)
    min_pop = st.slider("Minimal number of reviews", 0, 1000, 1, step=10)
    discovery = st.checkbox(
        "Discovery mode", value=True,
        help="Hides books from any series you have already rated (sequels, " \
             "prequels) and shows at most one book per series. Turn it off to see" 
             "them. Warning: based on community tags, errors can happen.")
    if ss.skipped_recs:
        st.caption(f"{len(ss.skipped_recs)} recommendation(s) skipped")
        if st.button("Show skipped books again", icon=":material/restart_alt:"):
            ss.skipped_recs = set()
            st.rerun()

### Main section ---------------------------------------------------------------
tab_reco, tab_model = st.tabs([
    "Recommendations", 
    "How it works"])

with tab_reco:  
    st.subheader(f"You should like... ")
    st.caption(f"Showing {max_top} books that are close to your annotations. "
            "Add annotations to improve the recommendations!")
    liked_ids = [r["book_id"] for r in labels if float(r["label"]) >= 0.75]

    # Actual training of model and fetching of results.
    recs = recommend(
        feats, 
        result, 
        {r["book_id"] for r in labels} | ss.skipped_recs, 
        top_n=max_top,
        min_popularity=min_pop, 
        liked_ids=liked_ids,
        discovery=discovery, 
        rated_ids=[r["book_id"] for r in labels])

    # Unlikely but just in case: what if all books are already rated?
    if not recs:
        st.info("No unrated book matches the current filters.")

    # For each book in recommendations, display a info card
    books = load_books()
    for rank, rec in enumerate(recs, start=1):
        book = books[rec["id"]]

        with st.container(border=True):
            left, right = st.columns([1, 5], vertical_alignment="center")

            # On the left side, a book's cover.
            with left:
                cover(book.get("cover_url"), book["title"])

            # On the right side all the information on the book.
            with right:
                # Informations
                st.markdown(f"#### {rank}. {book['title']}" + (f": {book['subtitle']}" if book.get("subtitle") else ""))
                meta = [book.get("author") or "Unknown author"]

                if book.get("first_publish_year"):
                    meta.append(str(book["first_publish_year"]))
                st.write(" · ".join(meta))
                if book.get("avg_rating"):
                    st.caption(f":material/star: {book['avg_rating']} avg · {book.get('n_ratings', 0):,} ratings on Babelio")
                st.progress(rec["proba"], text=f"Confidence: {rec['proba']:.0%} chance you'll like it")
                
                if rec["reasons"]:
                    st.caption("Why: " + " · ".join(f"`{r}`" for r in rec["reasons"]))
                
                if rec["similar_to"] and rec["similar_to"]["id"] in books:
                    sim = rec["similar_to"]
                    st.caption(f"Close to a book you liked: *{books[sim['id']]['title']}* "
                            f"(similarity {sim['similarity']:.0%})")
                
                with st.expander("Summary"):
                    st.write(book["summary"])

                # Actions
                st.caption("Already read it? Tell the model what you thought:")
                b1, b2, b3 = st.columns(3)
                b1.button("Liked", icon=":material/thumb_up:", key=f"rec_liked_{book['id']}",
                        on_click=rate_recommended, args=(book, 1.0), width="stretch")
                b2.button("Neutral", icon=":material/sentiment_neutral:", key=f"rec_meh_{book['id']}",
                        on_click=rate_recommended, args=(book, 0.5), width="stretch")
                b3.button("Disliked", icon=":material/thumb_down:", key=f"rec_disliked_{book['id']}",
                        on_click=rate_recommended, args=(book, 0.0), width="stretch")
                st.caption("Not read it, but are you tempted?")
                b4, b5, b6 = st.columns(3)
                b4.button("Interested", icon=":material/favorite:", key=f"rec_noint_{book['id']}",
                        on_click=interested, args=(book,), width="stretch",
                        help="Teaches the model that you like this kind of book.")
                b5.button("Not interested", icon=":material/block:", key=f"rec_int_{book['id']}",
                        on_click=not_interested, args=(book,), width="stretch",
                        help="Teaches the model that this kind of book is not for you.")
                b6.button("Skip", icon=":material/skip_next:", key=f"rec_skip_{book['id']}",
                        on_click=skip_recommended, args=(book["id"],), width="stretch",
                        help="Just hides it for this session; the model learns nothing.")


### Model summary --------------------------------------------------------------
with tab_model:
    st.subheader("Summary")
    st.markdown(
        "The recommendations are computed using a very basic Machine Learning "
        "model (a Logistic Regression) that is fitted on your annotations. "
        "It is method light enough to run fast on any computer. The model looks "
        "at the features of each book (author, summary, tags,...) and tries to " 
        "fit that to the book you like/dislike. It is mostly boring statistics... "
        "but what is important is that the data, model and recommendation are "
        "all local!\n\nThe tool is yours and, with a little help, you could tune  "
        "it yourself!")
    st.caption(
        "The following metrics are global scores that are useful to understand good "
        "(or bad) performances.")
    c1, c2, c3, c4 = st.columns(4)
    if result.cv_bal_acc is not None:
        c2.metric(
            "Balanced accuracy", 
            f"{result.cv_bal_acc:.0%}",
            help="The model ***predicts accurately*** what you might like/dislike in "
                f"about {result.cv_bal_acc:.0%} of cases (kind of like a weighted mean).",
            border=True)
        c3.metric(
            "ROC AUC", 
            f"{result.auc:.2f}",
            help="The model ***tells apart*** what you like/dislike in about "
                f"{100*result.auc:.0f}% of cases.",
            border=True)
    else:
        c2.metric(
            "Balanced accuracy (cross-validated)", 
            "n/a", 
            help="Needs at least 2 liked and 2 disliked books.",
            border=True)
        c3.metric(
            "ROC AUC", 
            "n/a",
            border=True)
    c1.metric(
        "Baseline score", 
        f"{BASELINE:.0%}",
        help="The model is only useful if balanced accuracy and ROC AUC beat this.",
        border=True)
    c4.metric(
        "Fitted on", 
        f"{result.n_pos + result.n_neg + result.n_neutral} labels",
        help=f"The annotations that you made: {result.n_pos} positive · {result.n_neg} negative · {result.n_neutral} neutral.",
        border=True)
    st.caption(
        f"Books of the same series count once in training so sequels do not "
        "inflate the score. The model is re-trained automatically after each "
        f"change, and saved in `{MODEL_DIR}/`.")


### Analyses section -----------------------------------------------------------
    st.divider()
    st.subheader("Prediction performance")
    st.text("This section is intented for data scientists who want to know a "
            "little more about the performances of the model, or for people "
            "curious enough to read through the numerous graphs and explanations.")
    render_analysis(result)