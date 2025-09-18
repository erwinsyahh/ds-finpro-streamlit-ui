import streamlit as st
import pandas as pd
import pickle
from utils.custom_transformers import DateTimeFeatures, TextCleaner, Preproccess
import os

st.set_page_config(page_title="Prediction", layout="wide")
st.title("🔮 Sentiment Prediction Interface")

# Load Model
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

with open(os.path.join(ARTIFACT_DIR, "finalized_model.pkl"), "rb") as f:
    ridge_model = pickle.load(f)

# Target variable
target = "sentiment_score"

st.write(f"Model is trained to predict **{target}**")

# ==========================================================
# 1. Single Record Prediction
# ==========================================================
st.subheader("Single Record Prediction")

# Example row to prefill
example_input = {
    'post_id': 'p123',
    'timestamp': '2025-09-05 12:34:00',
    'day_of_week': 'Friday',
    'platform': 'Twitter',
    'user_id': 'u456',
    'location': 'USA',
    'language': 'English',
    'text_content': 'Loving the new product launch!',
    'hashtags': '#excited',
    'mentions': '@brand',
    'keywords': 'launch,product',
    'topic_category': 'Tech',
    'sentiment_label': 'Positive',
    'emotion_type': 'Excited',
    'toxicity_score': 0.1,
    'likes_count': 50,
    'shares_count': 10,
    'comments_count': 5,
    'impressions': 1000,
    'engagement_rate': 0.05,
    'brand_name': 'BrandX',
    'product_name': 'GadgetY',
    'campaign_name': 'Launch2025',
    'campaign_phase': 'Launch',
    'user_past_sentiment_avg': 0.4,
    'user_engagement_growth': 0.2,
    'buzz_change_rate': 0.1,
}

user_input = {}
for f, default in example_input.items():
    if isinstance(default, (int, float)):
        val = st.number_input(f, value=float(default), format="%.6f")
    else:
        if f == 'text_content':
            val = st.text_area(f, value=str(default))
        else:
            val = st.text_input(f, value=str(default))
    user_input[f] = val

if st.button("Predict Single Record"):
    X_new = pd.DataFrame([user_input])
    try:
        preprocessor = Preproccess()

        X_transform = preprocessor.transform(X_new)

        st.dataframe(X_transform)

        pred = ridge_model.predict(X_transform)[0]
        st.success(f"Predicted {target}: {pred:.6f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==========================================================
# 2. Batch Prediction
# ==========================================================
st.write("---")
st.subheader("Batch Prediction")

st.markdown("Paste CSV rows (including header) to run batch predictions.")
st.caption("👇 Example CSV (you can edit and paste below):")

example_csv = """post_id,timestamp,day_of_week,platform,user_id,location,language,text_content,hashtags,mentions,keywords,topic_category,sentiment_label,emotion_type,toxicity_score,likes_count,shares_count,comments_count,impressions,engagement_rate,brand_name,product_name,campaign_name,campaign_phase,user_past_sentiment_avg,user_engagement_growth,buzz_change_rate
p123,2025-09-05 12:34:00,Friday,Twitter,u456,USA,English,"Loving the new product launch!","#excited","@brand","launch,product",Tech,Positive,Excited,0.1,50,10,5,1000,0.05,BrandX,GadgetY,Launch2025,Launch,0.4,0.2,0.1
"""

text = st.text_area("CSV input", example_csv, height=200)

if st.button("Predict Batch"):
    from io import StringIO
    try:
        df_new = pd.read_csv(StringIO(text))

        st.dataframe(df_new.head())

        preprocessor = Preproccess()
        df_transform = preprocessor.transform(df_new)

        st.dataframe(df_transform)
        preds = ridge_model.predict(df_transform)
        out = df_new.copy()
        out[f"pred_{target}"] = preds
        st.dataframe(out)
        st.success("✅ Batch predictions generated successfully")
    except Exception as e:
        st.error(f"Failed to parse CSV input: {e}")
