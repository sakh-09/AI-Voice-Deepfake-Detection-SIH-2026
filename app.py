import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import tempfile
import os

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Voice Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# PATHS
# ==========================================

MODEL_PATH = os.path.join(
    "models",
    "voice_deepfake_mfcc_model.keras"
)

SCALER_PATH = os.path.join(
    "models",
    "mfcc_feature_scaler.pkl"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler


try:

    model, scaler = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        f"❌ Model loading failed: {e}"
    )


# ==========================================
# TITLE
# ==========================================

st.title(
    "🛡️ AI Voice Deepfake Detection"
)

st.write(
    "AI-powered detection of synthetic and cloned voices"
)


if model_loaded:

    st.success(
        "✅ Model loaded successfully"
    )


# ==========================================
# MFCC FEATURE EXTRACTION
# ==========================================

def extract_mfcc_features(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mean_features = np.mean(
        mfcc,
        axis=1
    )

    std_features = np.std(
        mfcc,
        axis=1
    )

    features = np.concatenate(
        [
            mean_features,
            std_features
        ]
    )

    return features.reshape(1, -1)


# ==========================================
# REAL-TIME AUDIO
# ==========================================

st.header(
    "🎙️ Real-Time Voice Analysis"
)

st.write(
    "Speak into your microphone. "
    "The system will collect short audio segments "
    "and analyze them."
)


class AudioProcessor:

    def __init__(self):

        self.audio_frames = []


    def recv(self, frame):

        audio = frame.to_ndarray()

        self.audio_frames.append(
            audio
        )

        # Keep only recent frames
        if len(self.audio_frames) > 30:

            self.audio_frames.pop(0)

        return frame


ctx = webrtc_streamer(
    key="voice-analysis",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={
        "audio": True,
        "video": False
    },
    async_processing=True
)


# ==========================================
# INFORMATION
# ==========================================

st.divider()

st.subheader(
    "📊 Detection System"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "🎤 Audio Input\n\n"
        "Microphone audio"
    )

with col2:

    st.info(
        "🧠 AI Analysis\n\n"
        "MFCC + Deep Learning"
    )

with col3:

    st.info(
        "🚨 Risk Detection\n\n"
        "REAL / AI-generated"
    )


# ==========================================
# DEMO EXPLANATION
# ==========================================

st.divider()

st.subheader(
    "How it works"
)

st.write(
    """
    **Microphone → Audio Stream → MFCC Features → 
    Trained Deep Learning Model → AI Voice Probability → 
    Risk Assessment**
    """
)