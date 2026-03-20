# MLOps Mini Project — Tweet Emotion Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-orange.svg)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-remote%20tracking-green.svg)](https://dagshub.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **MLOps** project that demonstrates a complete machine-learning lifecycle for **tweet emotion classification** (happiness vs. sadness). The project covers data preprocessing, feature engineering, model training, experiment tracking with **MLflow** and **DagsHub**, and iterative model improvement through systematic hyperparameter tuning.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Technologies & Tools](#technologies--tools)
- [Project Structure](#project-structure)
- [Experiments Overview](#experiments-overview)
- [Installation & Setup](#installation--setup)
- [Running the Experiments](#running-the-experiments)
- [Experiment Tracking with MLflow & DagsHub](#experiment-tracking-with-mlflow--dagshub)
- [Results Summary](#results-summary)
- [Contributing](#contributing)
- [License](#license)

---

## Problem Statement

Social media platforms generate massive volumes of text expressing a wide range of human emotions. Automatically detecting the **emotional tone** of a tweet (e.g., *happiness* or *sadness*) has practical applications in mental health monitoring, brand sentiment analysis, and content moderation.

This project builds and iteratively improves a **binary text classifier** that predicts whether a tweet expresses happiness or sadness, while applying MLOps best practices — reproducible experiments, version-controlled models, and centralised metric tracking.

---

## Dataset

| Property      | Details |
|---------------|---------|
| **Source**    | [CampusX Jupyter Masterclass — Tweet Emotions CSV](https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv) |
| **Classes**   | `happiness` (label `1`) and `sadness` (label `0`) |
| **Features**  | Raw tweet text (`content` column) |
| **Task**      | Binary sentiment classification |

Raw tweets undergo the following text-normalisation pipeline before feature extraction:

1. Lower-casing
2. Stop-word removal (NLTK English stopwords)
3. Number removal
4. Punctuation removal
5. URL removal
6. Lemmatisation (WordNet Lemmatiser)

---

## Technologies & Tools

| Category | Tool / Library |
|---|---|
| Language | Python 3.8+ |
| Data manipulation | pandas, NumPy |
| NLP / preprocessing | NLTK (stopwords, WordNetLemmatizer) |
| Feature extraction | scikit-learn `CountVectorizer`, `TfidfVectorizer` |
| ML algorithms | scikit-learn `LogisticRegression`, `MultinomialNB`, `RandomForestClassifier`, `GradientBoostingClassifier`; XGBoost |
| Hyperparameter tuning | scikit-learn `GridSearchCV` |
| Experiment tracking | MLflow |
| Remote experiment store | DagsHub |
| Notebooks | Jupyter |
| Dependency management | pip / virtualenv / conda |
| Build automation | GNU Make |
| Code quality | flake8, tox |

---

## Project Structure

```
mlops-mini-project/
├── LICENSE
├── Makefile                    <- Automation commands: `make data`, `make train`, `make lint`
├── README.md                   <- This file
├── requirements.txt            <- Python dependencies
├── setup.py                    <- Makes the project pip-installable (`pip install -e .`)
├── test_environment.py         <- Validates the Python environment version
├── tox.ini                     <- tox configuration (flake8 settings)
│
├── data/                       <- (git-ignored) All data artefacts
│   ├── raw/                    <- Original, immutable data dump
│   ├── interim/                <- Intermediate transformed data
│   ├── processed/              <- Final, model-ready datasets
│   └── external/               <- Data from third-party sources
│
├── docs/                       <- Sphinx project documentation
│   ├── conf.py
│   ├── index.rst
│   └── getting-started.rst
│
├── models/                     <- Serialised models and model summaries
│
├── notebooks/                  <- Jupyter experiment notebooks
│   ├── dagshub_setup.py        <- DagsHub / MLflow initialisation helper
│   ├── exp1_baseline_model.ipynb        <- Experiment 1: BoW + Logistic Regression baseline
│   ├── exp2_BOW_VS_TFIDF.ipynb          <- Experiment 2: BoW vs TF-IDF across multiple classifiers
│   └── exp3_BOW&LOR.ipynb               <- Experiment 3: Logistic Regression hyperparameter tuning
│
├── references/                 <- Data dictionaries, papers, and explanatory materials
│
├── reports/                    <- Generated analysis reports (HTML, PDF, LaTeX)
│   └── figures/                <- Generated graphics and figures
│
└── src/                        <- Source code package
    ├── __init__.py
    ├── data/
    │   └── make_dataset.py     <- Downloads and prepares raw data
    ├── features/
    │   └── build_features.py   <- Transforms raw text into ML-ready features
    ├── models/
    │   ├── train_model.py      <- Trains and serialises the selected model
    │   └── predict_model.py    <- Loads a model and generates predictions
    └── visualization/
        └── visualize.py        <- Creates exploratory and results visualisations
```

---

## Experiments Overview

### Experiment 1 — Logistic Regression Baseline (`exp1_baseline_model.ipynb`)

Establishes a performance baseline using the simplest viable pipeline:

- **Vectorisation**: Bag-of-Words (`CountVectorizer`, max 1 000 features)
- **Classifier**: Logistic Regression (default hyperparameters)
- **Split**: 80 % train / 20 % test, `random_state=42`
- **Tracked metrics**: Accuracy, Precision, Recall, F1-score
- **Artefacts logged**: trained model (`.pkl`), vectoriser (`.pkl`), notebook

### Experiment 2 — BoW vs. TF-IDF (`exp2_BOW_VS_TFIDF.ipynb`)

Compares two feature-extraction strategies across five classifiers using a parent/child MLflow run structure:

| Vectoriser | Classifiers |
|---|---|
| Bag-of-Words (`CountVectorizer`) | Logistic Regression, Multinomial NB, XGBoost, Random Forest, Gradient Boosting |
| TF-IDF (`TfidfVectorizer`) | Logistic Regression, Multinomial NB, XGBoost, Random Forest, Gradient Boosting |

All 10 combinations are tracked as nested MLflow child runs under a single parent run named **"Bow vs TfIdf"**.

### Experiment 3 — Logistic Regression Hyperparameter Tuning (`exp3_BOW&LOR.ipynb`)

Applies `GridSearchCV` to find the optimal Logistic Regression configuration:

| Hyperparameter | Search Space |
|---|---|
| `C` (regularisation) | `[0.1, 1, 10]` |
| `penalty` | `['l1', 'l2']` |
| `solver` | `['liblinear']` |

The best estimator and its parameters are logged to MLflow under the run **"Logistic Regression Hyperparameter Tuning"**.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/DeepuML/Mlops-Mini-Project.git
cd Mlops-Mini-Project/mlops-mini-project
```

### 2. Create and activate a virtual environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
venv/Scripts/activate             # Windows

# Or using conda
conda create -n mlops-mini python=3.10
conda activate mlops-mini
```

### 3. Install dependencies

```bash
make requirements
# equivalent to: pip install -r requirements.txt
```

Additional libraries required by the notebooks (install once):

```bash
pip install mlflow dagshub scikit-learn xgboost nltk pandas numpy jupyter
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### 4. Configure DagsHub credentials

Export your DagsHub token so MLflow can authenticate:

```bash
export MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

Or set them inside a notebook before starting an MLflow run:

```python
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "<your-dagshub-username>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<your-dagshub-token>"
```

---

## Running the Experiments

### Via Jupyter Notebooks (recommended)

```bash
jupyter notebook
```

Open and run the notebooks in order:

1. `notebooks/exp1_baseline_model.ipynb`
2. `notebooks/exp2_BOW_VS_TFIDF.ipynb`
3. `notebooks/exp3_BOW&LOR.ipynb`

### Via Makefile

```bash
make data      # Download and prepare the dataset
make train     # Train the model (uses src/models/train_model.py)
make lint      # Run flake8 over src/
make clean     # Remove compiled Python artefacts
```

---

## Experiment Tracking with MLflow & DagsHub

All experiments are tracked on **DagsHub** at:

> **https://dagshub.com/DeepuML/Mlops-Mini-Project.mlflow**

Each notebook configures the remote tracker at the top:

```python
import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/DeepuML/Mlops-Mini-Project.mlflow")
dagshub.init(repo_owner='DeepuML', repo_name='Mlops-Mini-Project', mlflow=True)
```

**What is logged per run:**

| Category | Logged Items |
|---|---|
| Parameters | vectoriser type, number of features, test split size, model name, hyperparameters |
| Metrics | Accuracy, Precision, Recall, F1-score |
| Artefacts | Serialised model (`.pkl`), vectoriser (`.pkl`), source notebook |

To view experiments locally without DagsHub:

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

---

## Results Summary

| Experiment | Vectoriser | Classifier | Accuracy |
|---|---|---|---|
| Exp 1 — Baseline | BoW (1 000 features) | Logistic Regression | ~0.85 |
| Exp 2 — Best run | TF-IDF | Logistic Regression | ~0.87 |
| Exp 3 — Tuned | BoW (1 000 features) | Logistic Regression (`C=10`, `l2`) | ~0.86 |

> Exact metric values depend on the data download at runtime. Check the DagsHub MLflow UI for the authoritative results.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with a descriptive message: `git commit -m "Add: description of change"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request against `main`

Please make sure your code passes `flake8` linting (`make lint`) before submitting a PR.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
