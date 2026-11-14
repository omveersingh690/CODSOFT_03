
import os
import json
import html

import streamlit as st
import torch

from PIL import Image
from torchvision import transforms

from model import ImageCaptioningModel

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CaptionAI | Image Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_model/caption_model.pth"
VOCAB_PATH = "saved_model/vocab.json"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #080d1c,
        #111a35,
        #17102e
    );

    color: white;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hero section */

.hero {
    text-align: center;
    padding: 55px 15px 40px;
}

.hero-badge {
    display: inline-block;
    padding: 9px 18px;
    border-radius: 30px;

    background: rgba(139,92,246,0.15);

    border: 1px solid rgba(167,139,250,0.4);

    color: #c4b5fd;

    font-size: 13px;

    letter-spacing: 1px;
}

.hero h1 {
    font-size: clamp(36px, 6vw, 64px);
    font-weight: 800;
    color: white;
    margin-top: 22px;
    line-height: 1.2;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #a78bfa,
        #38bdf8
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #aab5d0;
    font-size: 17px;
    max-width: 680px;
    margin: auto;
    line-height: 1.7;
}

/* Cards */

.card {
    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.12);

    border-radius: 20px;

    padding: 24px;

    margin-top: 18px;
}

.card-title {
    color: #e9e7ff;
    font-size: 21px;
    font-weight: 700;
}

.card-description {
    color: #94a3b8;
    font-size: 14px;
}

/* Upload area */

[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.08);
    border: 2px dashed #6366f1;
    border-radius: 16px;
    padding: 15px;
}

/* Image */

[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* Buttons */

.stButton > button {
    width: 100%;

    background: linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 14px;

    font-weight: 700;

    font-size: 16px;

    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow: 0 8px 25px rgba(99,102,241,0.35);

    color: white;

    border: none;
}

/* Caption result */

.result-card {
    background: linear-gradient(
        135deg,
        rgba(124,58,237,0.20),
        rgba(37,99,235,0.15)
    );

    border: 1px solid rgba(139,92,246,0.4);

    border-radius: 18px;

    padding: 25px;

    margin-top: 20px;
}

.result-label {
    color: #c4b5fd;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}

.result-caption {
    color: white;
    font-size: 21px;
    line-height: 1.7;
    margin-top: 12px;
}

/* Footer */

.footer {
    text-align: center;

    padding: 35px 10px;

    margin-top: 50px;

    border-top: 1px solid rgba(255,255,255,0.12);

    color: #94a3b8;
}

/* Responsive */

@media(max-width:768px) {

    .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 30px 5px;
    }

    .hero p {
        font-size: 14px;
    }

    .card {
        padding: 17px;
    }

    .result-caption {
        font-size: 17px;
    }

}

</style>
""", unsafe_allow_html=True)

# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Trained model not found. Please run train.py first."
        )

    if not os.path.exists(VOCAB_PATH):

        raise FileNotFoundError(
            "Vocabulary file not found. Please run train.py first."
        )

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        vocab = json.load(file)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = ImageCaptioningModel(
        vocab_size=checkpoint["vocab_size"],
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model, vocab

# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        ✨ AI POWERED IMAGE INTELLIGENCE
    </div>

    <h1>
        Turn Images Into
        <span class="gradient-text">
            Stories.
        </span>
    </h1>

    <p>
        Discover the power of deep learning.
        Upload an image and let our ResNet50 + LSTM
        architecture transform visual content
        into meaningful descriptions.
    </p>

</div>
""", unsafe_allow_html=True)

# ==========================================
# FEATURE CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

features = [
    ("🧠", "ResNet50", "Extracts important visual features."),
    ("⚡", "LSTM Decoder", "Generates captions word by word."),
    ("✨", "AI Captioning", "Transforms images into descriptions.")
]

for col, feature in zip(
    [col1, col2, col3],
    features
):

    with col:

        st.markdown(
            f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:32px;">
                    {feature[0]}
                </div>

                <h3 style="color:#e9e7ff;">
                    {feature[1]}
                </h3>

                <p class="card-description">
                    {feature[2]}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# UPLOAD SECTION
# ==========================================

st.markdown("""
<div class="card">

    <div class="card-title">
        📤 Upload Your Image
    </div>

    <p class="card-description">
        Select a JPG, JPEG or PNG image
        to generate its AI description.
    </p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

# ==========================================
# IMAGE PROCESSING
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("""
    <div class="card-title">
        🖼️ Image Preview
    </div>
    """, unsafe_allow_html=True)

    image_col, details_col = st.columns(
        [1.5, 1]
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with details_col:

        st.markdown("""
        <div class="card">

            <h3 style="color:#c4b5fd;">
                Image Ready!
            </h3>

            <p class="card-description">
                Your image has been uploaded.
                The AI model is ready to analyze
                its visual content.
            </p>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.write("**File:**", uploaded_file.name)

        st.write(
            "**Dimensions:**",
            f"{image.width} × {image.height}"
        )

        st.write(
            "**Format:**",
            uploaded_file.type
        )

    st.write("")

    # ======================================
    # GENERATE CAPTION BUTTON
    # ======================================

    if st.button(
        "✨ Generate AI Caption",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🧠 ResNet50 is extracting image features..."
            ):

                model, vocab = load_model()

                image_tensor = transform(image)

                image_tensor = image_tensor.unsqueeze(0)

                image_tensor = image_tensor.to(DEVICE)

            with st.spinner(
                "✨ LSTM is generating your caption..."
            ):

                with torch.no_grad():

                    caption = model.generate(
                        image_tensor,
                        vocab
                    )

                st.session_state["generated_caption"] = caption

        except Exception as error:

            st.error(
                "Unable to generate caption."
            )

            st.exception(error)

    # ======================================
    # DISPLAY CAPTION
    # ======================================

    if st.session_state.get("generated_caption"):

        caption = st.session_state["generated_caption"]

        safe_caption = html.escape(caption)

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    ✨ AI GENERATED CAPTION
                </div>

                <div class="result-caption">
                    {safe_caption}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.text_area(
            "Copy your generated caption:",
            value=caption,
            height=90
        )

        if st.button("🔄 Generate Another Caption"):

            del st.session_state["generated_caption"]

            st.rerun()

else:

    st.markdown("""
    <div class="card" style="text-align:center;">

        <div style="font-size:55px;">
            🖼️
        </div>

        <h3 style="color:#c4b5fd;">
            Your Image Journey Starts Here
        </h3>

        <p class="card-description">
            Upload an image and experience
            AI-powered image captioning.
        </p>

    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">

    <h2 style="color:#c4b5fd;">
        ✨ CaptionAI
    </h2>

    <p>
        Powered by ResNet50 + LSTM
    </p>

    <p>
        Built using Python, PyTorch and Streamlit
    </p>

    <p>
        CODSOFT Artificial Intelligence Internship Project
    </p>

    <p>
        © 2026 CaptionAI
    </p>

</div>
""", unsafe_allow_html=True)
