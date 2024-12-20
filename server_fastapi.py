"""
API server for TTS
TODO: Integrate with server_editor.py?
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote
import boto3

import GPUtil  # type: ignore
import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from scipy.io import wavfile

from config import get_config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


config = get_config()
ln = config.server_config.language


# Start the pyopenjtalk_worker
# The pyopenjtalk_worker is a TCP socket server, so it is started here.
pyopenjtalk.initialize_worker()

# dict_data/ Apply the following dictionary data to pyopenjtalk.
update_dict()

# Preload BERT models/tokenizers.
# While it would automatically load when needed, it's better to preload for a better user experience since loading takes time.
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)
bert_models.load_model(Languages.EN)
bert_models.load_tokenizer(Languages.EN)
bert_models.load_model(Languages.ZH)
bert_models.load_tokenizer(Languages.ZH)


def raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )


class AudioResponse(Response):
    media_type = "audio/wav"


loaded_models: list[TTSModel] = []


def load_models(model_holder: TTSModelHolder):
    global loaded_models
    loaded_models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = TTSModel(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        # Skip loading all models at startup since it takes time and consumes too much memory.
        # model.load()
        loaded_models.append(model)


aws_key = config.server_config.aws_key
aws_secret = config.server_config.aws_secret
aws_region = config.server_config.aws_region
bucket_name = config.server_config.aws_bucket_name

s3Resource = boto3.resource('s3', region_name=aws_region, aws_access_key_id=aws_key,
                            aws_secret_access_key=aws_secret)

s3 = boto3.client('s3', region_name=aws_region, aws_access_key_id=aws_key,
                  aws_secret_access_key=aws_secret)


def downloadFile(s3Resource: Any, bucket_name: str, file_name: str, download_path: str):
    print(f"Downloading {file_name} from S3")
    s3Resource.Bucket(bucket_name).download_file(file_name, download_path)
    print("Download completed.")


def getS3ModelNames(s3: Any, bucket_name: str, folder: str):  # type: ignore
    models = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder)
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            model_name = key.split("/")[1]
            model_entry = next(
                (m for m in models if m["name"] == model_name), None)
            if model_entry is None:
                model_entry = {
                    "name": model_name,
                    "files": []
                }
                models.append(model_entry)

            model_entry["files"].append(key)

    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path(args.dir)
    model_holder = TTSModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)

    limit = config.server_config.limit
    if limit < 1:
        limit = None
    else:
        logger.info(
            f"The maximum length of the text is {limit}. If you want to change it, modify config.yml. Set limit to -1 to remove the limit."
        )
    app = FastAPI()
    allow_origins = config.server_config.origins
    if allow_origins:
        logger.warning(
            f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # app.logger = logger
    # This doesn't seem to be working. Not sure how to override the logger.

    @app.api_route("/voice", methods=["GET", "POST"], response_class=AudioResponse)
    async def voice(
        request: Request,
        text: str = Query(..., min_length=1,
                          max_length=limit, description="Text to speak"),
        encoding: str = Query(
            None, description="URL decode the text (ex, `utf-8`)"),
        model_name: str = Query(
            None,
            description="Model name (takes priority over model_id). Specify directory name in model_assets",
        ),
        model_id: int = Query(
            0, description="Model ID. Please specify the key value from `GET /models/info`"
        ),
        speaker_name: str = Query(
            None,
            description="Speaker name (takes priority over speaker_id). Specify the string from the second column of esd.list",
        ),
        speaker_id: int = Query(
            0, description="Speaker ID. Check spk2id in model_assets>[model]>config.json"
        ),
        sdp_ratio: float = Query(
            DEFAULT_SDP_RATIO,
            description="SDP(Stochastic Duration Predictor)/DP ratio. Higher ratio increases tone variation",
        ),
        noise: float = Query(
            DEFAULT_NOISE,
            description="Sample noise ratio. Higher values increase randomness",
        ),
        noisew: float = Query(
            DEFAULT_NOISEW,
            description="SDP noise. Higher values increase variation in pronunciation timing",
        ),
        length: float = Query(
            DEFAULT_LENGTH,
            description="Speech speed. Default is 1. Higher values make audio longer and speech slower",
        ),
        language: Languages = Query(
            ln, description="Language of the input text"),
        auto_split: bool = Query(
            DEFAULT_LINE_SPLIT, description="Split text by newlines when generating"),
        split_interval: float = Query(
            DEFAULT_SPLIT_INTERVAL, description="Length of silence (in seconds) between split segments"
        ),
        assist_text: Optional[str] = Query(
            None,
            description="Reference text to make the voice and emotion similar. Note that intonation and tempo may be affected",
        ),
        assist_text_weight: float = Query(
            DEFAULT_ASSIST_TEXT_WEIGHT, description="Strength of assist text"
        ),
        style: Optional[str] = Query(DEFAULT_STYLE, description="Style"),
        style_weight: float = Query(
            DEFAULT_STYLE_WEIGHT, description="Style strength"),
        reference_audio_path: Optional[str] = Query(
            None, description="Apply style from an audio file"
        ),
    ):
        """Infer text to speech (Generate emotional voice from text)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}" # type: ignore
        )
        if request.method == "GET":
            logger.warning(
                "The GET method is not recommended for this endpoint due to various restrictions. Please use the POST method."
            )
        if model_id >= len(
            model_holder.model_names
        ):  # Cannot use Query(le) because /models/refresh exists
            raise_validation_error(
                f"model_id={model_id} not found", "model_id")

        if model_name:
            # Note that the processing in load_models() ensures the validity of i
            model_ids = [i for i, x in enumerate(
                model_holder.models_info) if x.name == model_name]
            if not model_ids:
                raise_validation_error(
                    f"model_name={model_name} not found", "model_name"
                )
            # With the current implementation, directory names should not be duplicated...
            if len(model_ids) > 1:
                raise_validation_error(
                    f"model_name={model_name} is ambiguous", "model_name"
                )
            model_id = model_ids[0]

        model = loaded_models[model_id]
        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(
                    f"speaker_id={speaker_id} not found", "speaker_id"
                )
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(
                    f"speaker_name={speaker_name} not found", "speaker_name"
                )
            speaker_id = model.spk2id[speaker_name]
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        assert style is not None
        if encoding is not None:
            text = unquote(text, encoding=encoding)
        sr, audio = model.infer(
            text=text,
            language=language,
            speaker_id=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )
        logger.success("Audio data generated and sent successfully")
        with BytesIO() as wavContent:
            wavfile.write(wavContent, sr, audio)
            return Response(content=wavContent.getvalue(), media_type="audio/wav")

    @app.get("/models/info")
    def get_loaded_models_info():
        """Get information about loaded models"""

        result: dict[str, dict[str, Any]] = dict()
        for model_id, model in enumerate(loaded_models):
            result[str(model_id)] = {
                "config_path": model.config_path,
                "model_path": model.model_path,
                "device": model.device,
                "spk2id": model.spk2id,
                "id2spk": model.id2spk,
                "style2id": model.style2id,
            }
        return result

    @app.post("/models/refresh")
    def refresh():
        """Reload models when models are added/removed from the path"""
        model_holder.refresh()
        load_models(model_holder)
        return get_loaded_models_info()

    @app.get("/status")
    def get_status():
        """Get runtime environment status"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_total = memory_info.total
        memory_available = memory_info.available
        memory_used = memory_info.used
        memory_percent = memory_info.percent
        gpuInfo = []
        devices = ["cpu"]
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    @app.get("/tools/get_audio", response_class=AudioResponse)
    def get_audio(
        request: Request, path: str = Query(..., description="local wav path")
    ):
        """Get wav data"""
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}" # type: ignore
        )
        if not os.path.isfile(path):
            raise_validation_error(f"path={path} not found", "path")
        if not path.lower().endswith(".wav"):
            raise_validation_error(f"wav file not found in {path}", "path")
        return FileResponse(path=path, media_type="audio/wav")

    @app.get("/list_s3_model")
    def list_s3_model(request: Request,  # type: ignore
                      folder: str = Query("outputs", description='folder contain model name. ex: outputs')):
        """List model from S3"""
        models = getS3ModelNames(s3, bucket_name, folder)

        missing_models = []
        for model in models:
            model_path = os.path.join("model_assets", model["name"])
            if not os.path.exists(model_path):
                missing_models.append(model["name"])

        return {"modes": models, "missing": missing_models}

    @app.get("/sync_s3_model")
    def sync_s3_model(request: Request,
                      folder: str = Query("outputs", description='folder contain model name. ex: outputs')):
        """Sync model from S3"""
        models = getS3ModelNames(s3, bucket_name, folder)
        missing_models = []
        for model in models:
            model_path = os.path.join("model_assets", model["name"])
            if not os.path.exists(model_path):
                missing_models.append(model["name"])

        for model_name in missing_models:
            for model in models:
                if model["name"] == model_name:
                    for file in model["files"]:
                        os.makedirs(os.path.join("model_assets",
                                    model_name), exist_ok=True)
                        download_path = os.path.join(
                            "model_assets", model_name, file.split("/")[2])
                        downloadFile(s3Resource, bucket_name,
                                     file, download_path)

        model_holder.refresh()
        load_models(model_holder)
        return get_loaded_models_info()

    logger.info(f"server listen: http://127.0.0.1:{config.server_config.port}")
    logger.info(f"API docs: http://127.0.0.1:{config.server_config.port}/docs")
    logger.info(
        f"Input text length limit: {limit}. You can change it in server.limit in config.yml"
    )
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
