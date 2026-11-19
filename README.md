# Sentiment Analysis on Text

An educational sentiment classifier trained on NLTK's movie-review corpus, with an
interactive Streamlit interface for individual and batch predictions.

## Run the user interface

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads `sentiment_model.joblib` from the project directory. It can:

- classify English text as positive or negative;
- display the probability split and basic input statistics;
- compare up to 50 texts at once; and
- show the classifier's most informative learned features.

The model is trained on movie reviews, so its output is most useful for
review-style language. Confidence values are model estimates, not guarantees.
