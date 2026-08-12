
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

st.set_page_config(page_title="Fruit AI Classifier", page_icon="🍓", layout="wide")

@st.cache_resource
def load_model():
    classifier = tf.keras.models.load_model("fruit_classifier.keras")
    with open("class_labels.json") as f:
        classes = json.load(f)
    return classifier, classes

classifier, CLASSES = load_model()

st.title("🍎🍌🍊🍓 Fruit AI Classifier")
st.markdown("Upload any fruit photo — real-world backgrounds, any lighting, any angle. Trained with background-diversified augmentation for strong generalization.")

st.sidebar.header("Classes supported")
for c in CLASSES:
    st.sidebar.markdown(f"- {c}")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        img_resized = image.resize((160, 160))
        img_array = np.expand_dims(np.array(img_resized).astype(np.float32), axis=0)
        preds = classifier.predict(img_array, verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_class = CLASSES[pred_idx]
        confidence = preds[pred_idx] * 100

        st.markdown(f"### {pred_class}")
        st.progress(float(preds[pred_idx]))
        st.markdown(f"**Confidence: {confidence:.1f}%**")

        st.markdown("---")
        st.markdown("**All probabilities:**")
        for cls, prob in zip(CLASSES, preds):
            st.write(f"{cls}: {prob*100:.2f}%")
            st.progress(float(prob))
else:
    st.info("Upload an image to get started")

st.markdown("---")
st.caption("Built with EfficientNetV2 + background-diversified augmentation.")
