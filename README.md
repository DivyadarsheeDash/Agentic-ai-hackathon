# Agentic AI Hackathon — NLP Safety & Response Prototype

A notebook-based NLP prototype exploring how multiple lightweight language models can be combined to analyze incoming text and support an automated response workflow.

## Project Idea

The repository decomposes text handling into separate tasks rather than relying on one monolithic model:

```text
Incoming message
      ↓
Sentiment / negativity analysis
      ↓
Severity estimation
      ↓
Intent recognition
      ↓
Auto-reply component
```

The current repository should be viewed as an **experimental hackathon prototype**, not a production moderation or autonomous-agent system.

## Repository Components

### `sentiment_analysis_model.ipynb`

Builds a text sentiment classifier.

The notebook works with a dataset containing:

- `Sentence`
- `Sentiment`

and maps sentiment labels into numeric classes.

A TF-IDF representation is used and **Logistic Regression** is selected as the preferred model in the notebook.

### `severity_regression.ipynb`

Explores continuous severity estimation for harmful/negative text.

The notebook uses:

- TF-IDF features with up to 20,000 features
- unigrams and bigrams
- a continuous `severity` target
- Ridge regression as one of the fitted regressors

Rows without severity values are removed before model fitting.

### `intent recognition/`

Contains work related to classifying user intent.

### `auto_reply_generator.ipynb`

Explores automated reply categorization/generation logic.

The text pipeline uses TF-IDF with:

- English stop-word filtering
- `(1, 2)` n-grams
- maximum 5,000 features
- sublinear term frequency

A Logistic Regression classifier and grid search are used in the notebook workflow.

### Saved Artifacts

The repository also contains serialized model/vectorizer artifacts used by parts of the pipeline.

## Data

The project uses multiple text datasets.

The original README points to the **Jigsaw Toxic Comment Classification Challenge** data for the negativity/severity component:

https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Other sentiment data are prepared in the repository/notebook workflow.

## Tech Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- TF-IDF
- Logistic Regression
- Ridge Regression
- Joblib

## Why Modularize the NLP Tasks?

Different objectives require different signals:

- Sentiment answers *how positive/negative is the text?*
- Severity estimates *how serious is the harmful content?*
- Intent asks *what is the user trying to do?*
- Reply logic asks *what response category should be produced?*

Keeping these tasks separate makes each component easier to inspect and debug.

## Limitations

- The system is an experimental prototype.
- It should not be treated as a reliable real-world safety classifier without broader validation.
- Dataset domains differ, which can produce distribution shift.
- Automated replies require careful human-centered evaluation.
- No end-to-end production service is provided in the public repository.

## Future Work

- Define a single end-to-end inference interface.
- Add explicit metrics for every submodel.
- Add calibration and threshold analysis.
- Evaluate on out-of-domain conversations.
- Add bias/fairness checks.
- Add confidence-based abstention.
- Replace heuristic reply logic with a controlled, auditable response policy.

## Author

**Divyadarshee Dash**
