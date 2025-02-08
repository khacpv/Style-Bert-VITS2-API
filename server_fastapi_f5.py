from f5_tts_mlx.generate import generate
import boto3
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from config import get_config

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
    def f5_tts(request: Request, text: str = Query("", min_length=1, description="Text to speak"), model_name: str = Query("", description="Model name")):
        """inference f5-tts"""
        if (model_name == "taylor_4s_24kHz"):
            ref_audio_path = "inputs/taylor_4s_24kHz.wav"
            ref_audio_text = "If I had a really joyful experience making something. This is a pretty dark album"
            output_path = "tests/taylor_4s_24kHz.wav"
        else:
            raise HTTPException(status_code=400, detail="Invalid model name")

        generate(generation_text=text, ref_audio_path=ref_audio_path,
                 ref_audio_text=ref_audio_text, output_path=output_path)
        return FileResponse(output_path, media_type="audio/wav")

    uvicorn.run(
        app, port=3000, host="0.0.0.0", log_level="warning"
    )
