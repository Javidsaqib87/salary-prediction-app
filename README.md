# Salary Predictor

An end-to-end machine learning project that predicts a person's salary from their years of experience. It covers the full workflow: exploring and cleaning the data, training and evaluating a regression model, saving the model, and serving it through an interactive Streamlit app deployed on Streamlit Community Cloud.

**Live app:** https://YOUR-APP-NAME.streamlit.app  
**Repository:** https://github.com/Javidsaqib87/salary-prediction-app

![Screenshot of the deployed app](screenshots/app.png)

## What the app does

You enter a number of years of experience and the app returns an estimated salary. Alongside the prediction it shows:

- a likely low and high figure, based on the model's average error
- a chart of where the prediction sits against the real data
- a history of the predictions made in the session, which can be downloaded as a CSV
- a page for exploring the dataset
- a page explaining how well the model performs and what the numbers mean
- a warning when the input is outside the range the model was trained on

The app loads the saved model from `model/model.pkl`. It does not retrain anything when it runs.

## Dataset

- **Source:** salary dataset provided in class — [salary_dataset.csv on GitHub](https://raw.githubusercontent.com/SagarChhabriya/data-science/refs/heads/main/datasets/TBD/salary_dataset.csv)
- **Size:** 200 rows, 2 columns
- **Columns:** `Experience Years` (input) and `Salary` (target)

The data was already clean. I checked for missing values, duplicate rows, impossible values and outliers (1.5 × IQR rule) and found none, so all 200 rows were used. The only preprocessing step was renaming `Experience Years` to `YearsExperience` to make it easier to work with in code.

## Model

I used **simple linear regression** from scikit-learn.

The scatter plot of salary against experience is very close to a straight line (correlation 0.99), so a linear model was the natural starting point. To make sure I was not missing out on accuracy, I also tried a degree-2 polynomial regression and a decision tree under the same 5-fold cross-validation. Neither did better, so I kept the simplest model, which is also the easiest to explain:

```
Salary = 25,755 + 9,438 × Years of experience
```

In plain terms: an estimated starting salary of about 25,755, rising by about 9,438 for each additional year.

## Evaluation

The data was split 80/20 into training (160 rows) and test (40 rows) sets with `random_state=42`.

| Metric | Training set | Test set |
| --- | --- | --- |
| R² | 0.979 | 0.983 |
| MAE | 4,708 | 4,056 |
| RMSE | 6,010 | 4,873 |

5-fold cross-validated R² (shuffled folds): **0.980**

The training and test scores are close, so the model is not overfitting, and the cross-validation result shows the test score is not down to a lucky split. On data it has not seen, the model explains about 98% of the variation in salary and is off by roughly 4,000 on average.

One thing I ran into: the CSV is sorted by experience, so cross-validation has to shuffle the rows. Without shuffling, each fold only contains a narrow band of experience levels and the scores come out misleadingly low.

## Project structure

```
salary-prediction-app/
├── data/
│   └── salary_dataset.csv        # the dataset
├── model/
│   ├── model.pkl                 # trained model (joblib)
│   └── metrics.json              # evaluation results shown in the app
├── notebooks/
│   └── model_training.ipynb      # exploration, training and evaluation
├── screenshots/
│   └── app.png                   # screenshot of the deployed app
├── app.py                        # Streamlit application
├── train_model.py                # script version of the training steps
├── requirements.txt
├── README.md
└── .gitignore
```

## Running it locally

You need Python 3.11 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/salary-prediction-app.git
cd salary-prediction-app

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

### Retraining the model

The trained model is already in the repository, so this step is optional. To rebuild it:

```bash
python train_model.py
```

or open `notebooks/model_training.ipynb` (`pip install notebook` first) and run all cells. Both produce the same `model/model.pkl`.

## Deployment

The app is deployed on Streamlit Community Cloud straight from this repository, with `app.py` as the entry point and the dependencies installed from `requirements.txt`. Every push to the `main` branch redeploys it automatically.

## Limitations

- Experience is the only input. Real salaries also depend on role, location, industry and education.
- The model was trained on 0.6 to 14.8 years of experience. Outside that range it simply extends the same straight line, which may not reflect reality.
- With 200 records this is a learning project, not a salary benchmark.

## Tools used

Python, pandas, NumPy, scikit-learn, joblib, Matplotlib, Streamlit, Git and GitHub.

## Author
javid Iqbal Saqib
