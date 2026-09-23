# 📰 Fake News Detector

Corrected Streamlit + Machine Learning project.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit Cloud
Upload the project to GitHub and deploy `app.py` using Streamlit Community Cloud. Keep `requirements.txt` in the repository root.

## Dataset
The project includes a small demo `train.csv`, so it runs without downloading any external dataset. For a better model, replace it with a larger compatible dataset containing a text column (`text`, `content`, `article`, or `title`) and a binary label column (`label`, `class`, or `target`) with labels 0 and 1.

## Important
This is an educational ML classifier. Its prediction is not proof that a news story is true or false.
