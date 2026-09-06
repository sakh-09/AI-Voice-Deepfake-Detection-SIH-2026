import os
import numpy as np
import librosa
import tensorflow as tf
import joblib


# ==============================
# MODEL PATHS
# ==============================

MODEL_PATH = os.path.join(
    "models",
    "voice_deepfake_mfcc_model.keras"
)

SCALER_PATH = os.path.join(
    "models",
    "mfcc_feature_scaler.pkl"
)


# ==============================
# LOAD MODEL
# ==============================

loaded_model = tf.keras.models.load_model(
    MODEL_PATH
)

loaded_scaler = joblib.load(
    SCALER_PATH
)


# ==============================
# VOICE PREDICTION
# ==============================

def predict_voice(audio_file):

    audio, sr = librosa.load(
        audio_file,
        sr=16000,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    features_scaled = loaded_scaler.transform(
        features.reshape(1, -1)
    )

    fake_probability = float(
        loaded_model.predict(
            features_scaled,
            verbose=0
        )[0][0]
    )

    real_probability = 1 - fake_probability

    if fake_probability >= 0.5:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    risk_score = fake_probability * 100

    if risk_score >= 80:
        risk_level = "HIGH"
    elif risk_score >= 50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "prediction": prediction,
        "fake_probability": round(fake_probability * 100, 2),
        "real_probability": round(real_probability * 100, 2),
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level
    }


# ==============================
# PREVENTION RECOMMENDATION
# ==============================

def get_recommendation(risk_level):

    if risk_level == "HIGH":

        return {
            "action": "DO NOT TRUST THE VOICE ALONE",

            "recommendation":
                "AI-generated or cloned voice is highly likely. "
                "Do not approve sensitive requests based only on this voice.",

            "verification": [
                "Perform an independent callback",
                "Use multi-factor authentication",
                "Escalate to a supervisor if required"
            ]
        }

    elif risk_level == "MEDIUM":

        return {
            "action": "ADDITIONAL VERIFICATION REQUIRED",

            "recommendation":
                "The voice shows a moderate probability of being AI-generated. "
                "Perform secondary verification before sensitive actions.",

            "verification": [
                "Perform an independent callback",
                "Verify the caller through another trusted channel"
            ]
        }

    else:

        return {
            "action": "LOW RISK",

            "recommendation":
                "No strong indication of an AI-generated voice was detected.",

            "verification": [
                "Continue normal verification procedures"
            ]
        }


# ==============================
# COMPLETE ANALYSIS
# ==============================

def analyze_voice(audio_file):

    result = predict_voice(audio_file)

    recommendation = get_recommendation(
        result["risk_level"]
    )

    return {
        "prediction": result["prediction"],
        "fake_probability": result["fake_probability"],
        "real_probability": result["real_probability"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "action": recommendation["action"],
        "recommendation": recommendation["recommendation"],
        "verification": recommendation["verification"]
    }