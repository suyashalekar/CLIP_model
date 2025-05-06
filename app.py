import streamlit as st
import torch
from PIL import Image
import open_clip
import pandas as pd

@st.cache_resource
def load_model():
    model, preprocess, tokenizer = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, preprocess, tokenizer, device

# Load model once
model, preprocess, tokenizer, device = load_model()

# App title
st.title("🧠 Iris Health Prediction App (CLIP-based)")

# Image upload
uploaded_file = st.file_uploader("📷 Upload an Iris Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Iris Image", use_column_width=True)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Health prompts (load from another file or define separately)
    class_descriptions = [
    # 🔵 Nervous System
    "An iris with a distorted pupil indicating neurological imbalance",
    "An iris with concentric stress rings indicating chronic anxiety or emotional overload",
    "An iris with signs of mental fatigue and tight fiber structure",
    "An iris showing radial furrows and stress lines from chronic emotional pressure",

    # 🟢 Digestive System
    "An iris with signs of sluggish colon and digestive insufficiency",
    "An iris with a pale pancreas zone and faded stomach ring",
    "An iris indicating bloating or gas from transverse colon stress",
    "An iris showing signs of poor nutrient absorption in the small intestine",

    # 🔴 Circulatory System
    "An iris with a visible cholesterol ring and signs of arteriosclerosis",
    "An iris showing blue or grey rings indicating low oxygenation",
    "An iris with signs of poor circulation and lymph congestion",
    "An iris showing weak hemoglobin absorption or low iron levels",

    # 🟡 Endocrine System
    "An iris showing thyroid imbalance affecting metabolism or mood",
    "An iris with signs of adrenal exhaustion and fatigue",
    "An iris with pigmentation in the ovarian zone indicating hormonal imbalance",
    "An iris with signs of insulin resistance and pancreas weakness",
    "An iris showing pituitary stress affecting hormonal regulation",

    # 🟣 Musculoskeletal System
    "An iris showing signs of weakness in spine, legs, or joint areas",
    "An iris showing pigmentation near eye area indicating neck, spine, or posture tension",

    # 🟠 Lymphatic / Immune System
    "An iris showing lymphatic congestion in the lymphatic rosary",
    "An iris with signs of immune overload from toxin buildup",

    # 🌬️ Respiratory System
    "An iris showing pigmentation in the lung zone suggesting toxin buildup or shallow breathing",

    # 💧 Urinary System
    "An iris with irritation markers near the bladder or urethral zone",

    # ⚪ Vitality & General
    "An iris with dark radial pigments suggesting toxin accumulation near pupil",
    "An iris showing overall low vitality and burnout",
    "A healthy iris with clear fibers and no stress rings",
]


    # Tokenize and encode
    text_inputs = tokenizer(class_descriptions).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T)
        probs = similarity.softmax(dim=-1)[0]

    # Sort top 5 results
    topk = 5
    top_probs, top_indices = probs.topk(topk)
    top_results = [(class_descriptions[i], top_probs[j].item()) for j, i in enumerate(top_indices)]

    # Display predictions
    st.markdown("### 🔍 Top 5 Health Pattern Predictions:")
    for desc, prob in top_results:
        st.markdown(f"- **{desc}**: {prob * 100:.2f}% confidence")

