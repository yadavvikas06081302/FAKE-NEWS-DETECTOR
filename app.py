import os, re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.caption("Machine-learning based educational demo")

DEMO_DATA = [('Government announces new public transport timetable', 0), ('Local university opens registration for the new academic session', 0), ('Scientists publish a peer reviewed study on renewable energy', 0), ('Weather department issues an official rainfall advisory', 0), ('City hospital opens a new emergency care unit', 0), ('Election commission publishes official voter information', 0), ('National bank announces revised service hours', 0), ('School releases the official examination schedule', 0), ('Researchers report results from a controlled laboratory study', 0), ('Transport department publishes road safety guidelines', 0), ('University confirms scholarship applications will open', 0), ('Health ministry publishes official vaccination information', 0), ('Municipal corporation announces waste collection schedule', 0), ('Railway authority releases an official passenger advisory', 0), ('Police department posts a traffic diversion notice', 0), ('Space agency publishes an official mission update', 0), ('Experts publish a report after reviewing available evidence', 0), ('The court publishes its order on the official website', 0), ('Government website lists the public service application process', 0), ('University researchers publish findings in a scientific journal', 0), ('Scientists secretly discover a miracle pill that cures every disease overnight', 1), ('Forward this message immediately or your bank account will be closed today', 1), ('Aliens have landed and the government is hiding the evidence', 1), ('A magic fruit guarantees instant weight loss without diet or exercise', 1), ('Secret celebrity video proves that every news channel is lying', 1), ('Click this link now to receive free money from the government', 1), ('Doctors are shocked by this one simple trick that reverses all illness', 1), ('A hidden law will make every citizen rich next week', 1), ('Anonymous source claims the moon will disappear tomorrow', 1), ('Share this post with ten people or your phone will stop working', 1), ('A miracle treatment works for every patient with zero side effects', 1), ('Breaking secret report says all schools will permanently close tomorrow', 1), ('Viral message claims a household item can replace every medicine', 1), ('Internet users say a mysterious signal proves aliens control the weather', 1), ('Secret government document predicts instant wealth for everyone', 1), ('A famous actor supposedly reveals a secret cure in a viral video', 1), ('Forwarded message says banks will give free cash to everyone tonight', 1), ('Unverified post claims a single food can prevent every infection', 1), ('Viral social media post says one drink removes all toxins', 1), ('Anonymous post claims tomorrow is an official nationwide holiday', 1)]

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\\s]", " ", str(text)).lower()
    return re.sub(r"\\s+", " ", text).strip()

@st.cache_resource
def train_model():
    path = os.path.join(os.path.dirname(__file__), "train.csv")
    data = None
    source = "built-in demo dataset"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path).fillna("")
            text_col = next((c for c in ["text","content","article","title"] if c in df.columns), None)
            label_col = next((c for c in ["label","class","target"] if c in df.columns), None)
            if text_col and label_col:
                data = pd.DataFrame({"text":df[text_col].astype(str),
                                     "label":pd.to_numeric(df[label_col], errors="coerce")}).dropna()
                data["label"] = data["label"].astype(int)
                data = data[data["label"].isin([0,1])]
                if len(data) >= 10 and data["label"].nunique() == 2:
                    source = f"train.csv ({len(data):,} rows)"
                else:
                    data = None
        except Exception:
            data = None
    if data is None:
        data = pd.DataFrame(DEMO_DATA, columns=["text","label"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2),
                                  max_features=30000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(data["text"].map(clean_text), data["label"])
    return model, source, len(data)

model, source, rows = train_model()
st.success(f"Model ready • {source} • {rows:,} examples")

text = st.text_area("Paste a news headline or article", height=180,
                    placeholder="Enter news text here...")
if st.button("🔍 Check News", use_container_width=True):
    if not text.strip():
        st.error("Please enter a news headline or article.")
    else:
        probs = model.predict_proba([clean_text(text)])[0]
        pred = int(model.predict([clean_text(text)])[0])
        conf = max(probs)*100
        if pred == 1:
            st.error("⚠️ Prediction: Potentially Fake / Unreliable")
        else:
            st.success("✅ Prediction: Potentially Real / Reliable")
        st.metric("Model confidence", f"{conf:.1f}%")
        st.info("This is an educational classifier, not proof that a story is true or false. Verify important claims with reliable sources.")

with st.expander("📁 train.csv format"):
    st.code('text,label\n"Official government announcement...",0\n"Miracle cure claim...",1')
