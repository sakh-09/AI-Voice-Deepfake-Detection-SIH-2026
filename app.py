from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from predict import analyze_voice


app = FastAPI(
    title="AI Voice Deepfake Detection API",
    description="API for AI-generated voice detection and risk assessment",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "AI Voice Deepfake Detection API"
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    # Create temporary audio file
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        contents = await file.read()
        temp.write(contents)
        temp_path = temp.name

    try:

        # Run your existing AI pipeline
        result = analyze_voice(temp_path)

        return result

    finally:

        # Delete temporary audio
        if os.path.exists(temp_path):
            os.remove(temp_path)