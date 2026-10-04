"""
Salary Predictor - Streamlit app

Loads the linear regression model trained in notebooks/model_training.ipynb
and predicts a salary from years of experience. The model is loaded from
disk once and reused; nothing is retrained when the app runs.

Run locally with:  streamlit run app.py
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "salary_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

FEATURE = "YearsExperience"
TARGET = "Salary"
MAX_INPUT_YEARS = 50.0

BLUE = "#4C72B0"
ORANGE = "#DD8452"

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the serialized model once and keep it in memory."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    """Load the dataset with the same column names used during training."""
    df = pd.read_csv(DATA_PATH)
    return df.rename(columns={"Experience Years": FEATURE})


@st.cache_data
def load_metrics():
    """Load the evaluation results saved by the training notebook."""
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict_salary(model, years):
    """Return the predicted salary for a given number of years."""
    features = pd.DataFrame({FEATURE: [float(years)]})
    return float(model.predict(features)[0])


def money(value):
    """Format a salary figure with thousands separators."""
    return f"{value:,.0f}"


# --------------------------------------------------------------------------
# Load everything the app needs
# --------------------------------------------------------------------------
if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        "The trained model could not be found. Run `python train_model.py` "
        "(or the notebook in `notebooks/`) to create `model/model.pkl` first."
    )
    st.stop()

model = load_model()
df = load_data()
metrics = load_metrics()

exp_min = metrics["experience_min"]
exp_max = metrics["experience_max"]
test_mae = metrics["test"]["mae"]

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About this app")
    st.write(
        "A small end-to-end machine learning project: a regression model "
        "trained on a salary dataset, served through Streamlit."
    )
    st.subheader("Model at a glance")
    st.write("**Algorithm:** Linear Regression")
    st.write(f"**Trained on:** {metrics['train_rows']} records")
    st.write(f"**Tested on:** {metrics['test_rows']} records")
    st.write(f"**Test R²:** {metrics['test']['r2']:.3f}")
    st.subheader("Dataset")
    st.write(
        "Salary dataset shared in class "
        "([source on GitHub](https://github.com/SagarChhabriya/data-science/"
        "tree/main/datasets/TBD))."
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("💼 Salary Predictor")
st.write(
    "Enter the number of years someone has worked and the app estimates "
    "their salary. The estimate comes from a linear regression model "
    "trained on 200 salary records."
)

tab_predict, tab_data, tab_model, tab_about = st.tabs(
    ["Predict", "Explore the data", "Model performance", "How it works"]
)


# --------------------------------------------------------------------------
# Tab 1: Prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Estimate a salary")

    years = st.number_input(
        "Years of experience",
        min_value=0.0,
        max_value=MAX_INPUT_YEARS,
        value=5.0,
        step=0.5,
        format="%.1f",
        help=f"The model was trained on people with {exp_min} to {exp_max} years of experience.",
    )

    # Input validation: the model is only reliable inside the range it has seen.
    outside_range = years < exp_min or years > exp_max
    if outside_range:
        st.warning(
            f"{years:.1f} years is outside the range the model was trained on "
            f"({exp_min} to {exp_max} years). You will still get a number, but "
            "treat it as a rough extrapolation."
        )

    if st.button("Predict salary", type="primary"):
        prediction = predict_salary(model, years)
        st.session_state.last = {"years": float(years), "salary": prediction}
        st.session_state.history.append(
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Years of experience": float(years),
                "Predicted salary": round(prediction),
                "Within training range": "No" if outside_range else "Yes",
            }
        )

    last = st.session_state.get("last")
    if last:
        st.success(
            f"Estimated salary for **{last['years']:.1f} years** of experience: "
            f"**{money(last['salary'])}**"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted salary", money(last["salary"]))
        col2.metric("Likely low end", money(max(last["salary"] - test_mae, 0)))
        col3.metric("Likely high end", money(last["salary"] + test_mae))
        st.caption(
            f"The low and high figures are the prediction plus or minus the model's "
            f"average error on unseen data (about {money(test_mae)}). Salaries are in "
            "the same unit as the dataset."
        )

        # Show where the prediction sits against the real data.
        x_max = max(exp_max, last["years"])
        line_x = pd.DataFrame({FEATURE: np.linspace(0, x_max, 100)})
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df[FEATURE], df[TARGET], alpha=0.45, color=BLUE, label="Dataset")
        ax.plot(line_x[FEATURE], model.predict(line_x), color="black", linewidth=1.8, label="Model")
        ax.scatter(
            [last["years"]], [last["salary"]],
            color=ORANGE, s=160, zorder=5, edgecolor="black", label="Your prediction",
        )
        ax.set_xlabel("Years of experience")
        ax.set_ylabel("Salary")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Choose a number of years and press **Predict salary**.")

    # Prediction history for this session.
    if st.session_state.history:
        st.divider()
        st.subheader("Prediction history")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df.iloc[::-1], hide_index=True)

        col_a, col_b = st.columns(2)
        col_a.download_button(
            "Download history as CSV",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
        if col_b.button("Clear history"):
            st.session_state.history = []
            st.session_state.pop("last", None)
            st.rerun()


# --------------------------------------------------------------------------
# Tab 2: Dataset
# --------------------------------------------------------------------------
with tab_data:
    st.subheader("The dataset")
    st.write(
        "Each row is one person: how many years they have worked and what they earn. "
        "The data had no missing values, duplicates or outliers, so all rows were used."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(df)}")
    col2.metric("Experience range", f"{df[FEATURE].min():.1f} to {df[FEATURE].max():.1f} yrs")
    col3.metric("Average salary", money(df[TARGET].mean()))

    st.markdown("**Salary against experience**")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df[FEATURE], df[TARGET], alpha=0.6, color=BLUE)
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Salary")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"The points sit close to a straight line (correlation "
        f"{df[FEATURE].corr(df[TARGET]):.2f}), which is why a linear model fits well."
    )

    st.markdown("**How the values are spread**")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(df[FEATURE], bins=15, color=BLUE, edgecolor="white")
    axes[0].set_xlabel("Years of experience")
    axes[0].set_ylabel("People")
    axes[1].hist(df[TARGET], bins=15, color="#55A868", edgecolor="white")
    axes[1].set_xlabel("Salary")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Summary statistics"):
        st.dataframe(df.describe().round(2))

    with st.expander("View the raw data"):
        st.dataframe(df, hide_index=True)


# --------------------------------------------------------------------------
# Tab 3: Model performance
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("How good is the model?")
    st.write(
        f"The data was split 80/20. The model learned from {metrics['train_rows']} records "
        f"and was then checked on {metrics['test_rows']} records it had never seen. "
        "The figures below are from that unseen test set."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("R² score", f"{metrics['test']['r2']:.3f}")
    col2.metric("Mean absolute error", money(metrics["test"]["mae"]))
    col3.metric("Root mean squared error", money(metrics["test"]["rmse"]))

    st.markdown(
        f"""
- **R² of {metrics['test']['r2']:.3f}** means experience alone explains about
  {metrics['test']['r2'] * 100:.0f}% of the differences in salary.
- **MAE of {money(metrics['test']['mae'])}** means a typical prediction is off by
  roughly that amount, in either direction.
- **RMSE of {money(metrics['test']['rmse'])}** is a similar measure that gives more
  weight to the larger misses.
"""
    )

    st.markdown("**Training vs. test scores**")
    scores = pd.DataFrame(
        {
            "Set": ["Training", "Test"],
            "R²": [round(metrics["train"]["r2"], 3), round(metrics["test"]["r2"], 3)],
            "MAE": [round(metrics["train"]["mae"]), round(metrics["test"]["mae"])],
            "RMSE": [round(metrics["train"]["rmse"]), round(metrics["test"]["rmse"])],
        }
    )
    st.dataframe(scores, hide_index=True)
    st.caption(
        "The two rows are close to each other, so the model is not overfitting. "
        f"5-fold cross-validation gives an average R² of {metrics['cv_r2_mean']:.3f}, "
        "which confirms the result does not depend on one particular split."
    )


# --------------------------------------------------------------------------
# Tab 4: Explanation
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("How the prediction is made")
    st.write(
        "Linear regression draws the straight line that best fits the data. "
        "Once the line is known, a prediction is just a matter of reading off the "
        "salary at a given number of years."
    )
    st.latex(
        rf"\text{{Salary}} = {metrics['intercept']:,.0f} + "
        rf"{metrics['slope']:,.0f} \times \text{{Years of experience}}".replace(",", r"{,}")
    )
    st.markdown(
        f"""
- **{money(metrics['intercept'])}** is the estimated starting salary with no experience.
- **{money(metrics['slope'])}** is how much the salary rises, on average, for each extra year.
"""
    )

    st.subheader("Things to keep in mind")
    st.markdown(
        f"""
- Experience is the only input. Role, location, industry and education all affect real
  salaries and are not part of this model.
- The model has only seen people with {exp_min} to {exp_max} years of experience.
  Predictions outside that range assume the same straight line carries on, which may not hold.
- The dataset has 200 records, so this is a learning project and not a salary benchmark.
"""
    )

    st.subheader("Project workflow")
    st.markdown(
        """
1. Load and explore the dataset
2. Check for missing values, duplicates and outliers
3. Split into training and test sets
4. Train a linear regression model
5. Evaluate with R², MAE and RMSE
6. Save the model with joblib
7. Load the saved model in this app and serve predictions
"""
    )
