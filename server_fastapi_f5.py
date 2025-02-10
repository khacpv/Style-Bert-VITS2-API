from f5_tts_mlx.generate import generate
import boto3
from fastapi import FastAPI, Query, Request, UploadFile, File, HTTPException
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from config import get_config
from datetime import datetime
import subprocess
import os

config = get_config()

aws_key = config.server_config.aws_key
aws_secret = config.server_config.aws_secret
aws_region = config.server_config.aws_region
bucket_name = config.server_config.aws_bucket_name

s3 = boto3.client('s3', region_name=aws_region, aws_access_key_id=aws_key,
                  aws_secret_access_key=aws_secret)

s3Resource = boto3.resource('s3', region_name=aws_region, aws_access_key_id=aws_key,
                            aws_secret_access_key=aws_secret)

# ffmpeg -i inputs/ken.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 inputs/ken_24kHz.wav

# generate(generation_text="If you want to use your own reference audio sample, make sure it's a mono, 24kHz wav file of around 5-10 seconds.",
#          ref_audio_path="inputs/taylor_4s_24kHz.wav",
#          ref_audio_text="If I had a really joyful experience making something. This is a pretty dark album",
#          output_path="tests/taylor_4s_24kHz.wav")

MODEL_DIR = "model_assets_f5"
MODEL_TMP_DIR = "model_tmp_f5"
models = []

if not os.path.exists(MODEL_TMP_DIR):
    os.makedirs(MODEL_TMP_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def get_model_names():
    """get model names"""
    global models
    models = [f[:-4] for f in os.listdir(MODEL_DIR) if f.endswith('.wav')]
    return models


if __name__ == "__main__":
    app = FastAPI()
    allow_origins = "*"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/f5-tts")
    def f5_tts(request: Request, text: str = Query("", min_length=1, description="Text to speak"), model_name: str = Query("ken_24kHz", description="Model name")):
        """inference f5-tts"""
        ref_audio_path = f"{MODEL_DIR}/{model_name}.wav"
        ref_audio_text = open(f"{MODEL_DIR}/{model_name}.txt", "r").read()
        output_path = f"{MODEL_TMP_DIR}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        generate(generation_text=text, ref_audio_path=ref_audio_path,
                 ref_audio_text=ref_audio_text, output_path=output_path)
        return FileResponse(output_path, media_type="audio/wav")

    @app.get("/f5-tts/models")
    def f5_tts_list_models():
        """list models"""
        models = get_model_names()
        return {"models": models}

    @app.post('/f5-tts/model')
    async def f5_tts_model(model_name: str = Query(..., description="Model name"),
                           sample_text: str = Query(...,
                                                    description="Sample text"),
                           audio_file: UploadFile = File(...)):
        """create model"""

        # Validate model_name
        if not re.match(r'^[a-zA-Z0-9]+$', model_name):
            raise HTTPException(
                status_code=400, detail="Model name can only contain alphanumeric characters (a-z, A-Z, 0-9) and cannot contain spaces or special characters.")

        # Validate sample_text
        if not sample_text or sample_text.strip() == "":
            raise HTTPException(
                status_code=400, detail="Sample text cannot be empty or null.")

        # Validate audio_file
        if audio_file.content_type != 'audio/wav':
            raise HTTPException(
                status_code=400, detail="Uploaded file must be a WAV file.")

        model_path = f"{MODEL_TMP_DIR}/{model_name}.wav"
        model_text_path = f"{MODEL_DIR}/{model_name}.txt"

        # Save the uploaded audio file
        with open(model_path, "wb") as buffer:
            buffer.write(await audio_file.read())

        # Save the sample text to a file
        with open(model_text_path, "w") as text_file:
            text_file.write(sample_text)

        command = [
            'ffmpeg',
            '-i', model_path,
            '-ac', '1',
            '-ar', '24000',
            '-sample_fmt', 's16',
            '-t', '10',
            f"{MODEL_DIR}/{model_name}.wav"
        ]

        subprocess.run(command, check=True)
        os.remove(model_path)
        return {"message": f"Model {model_name} created successfully."}

    uvicorn.run(
        app, port=3000, host="0.0.0.0", log_level="warning"
    )
