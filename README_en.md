# Style-Bert-VITS2

**Please be sure to read the [request and default model usage agreement](/docs/TERMS_OF_USE.md) before using.**

Bert-VITS2 with more controllable voice styles.

https://github.com/litagin02/Style-Bert-VITS2/assets/139731664/e853f9a2-db4a-4202-a1dd-56ded3c562a0

You can install via `pip install style-bert-vits2` (inference only), see [library.ipynb](/library.ipynb) for example usage.

- **Explanatory tutorial video** [YouTube](https://youtu.be/aTUSzgDl1iY)　[Nico Nico Douga](https://www.nicovideo.jp/watch/sm43391524)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- [**Frequently Asked Questions** (FAQ)](/docs/FAQ.md)
- [🤗 Online demo is here](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-Editor-Demo)
- [Zenn's explanatory article](https://zenn.dev/litagin/articles/034819a5256ff4)

- [**Release page**](https://github.com/litagin02/Style-Bert-VITS2/releases/)、[Update history](/docs/CHANGELOG.md)
  - 2024-09-09: Ver 2.6.1: Only bug fixes such as not being able to learn well in Google colab
  - 2024-06-16: Ver 2.6.0 (Addition of model differential merge, weighted merge, null model merge, see [this article](https://zenn.dev/litagin/articles/1297b1dc7bdc79) for usage)
  - 2024-06-14: Ver 2.5.1 (Only changed the usage agreement to request)
  - 2024-06-02: Ver 2.5.0 (**[Addition of usage agreement](/docs/TERMS_OF_USE.md)**, style generation from folder separation, addition of Koharu Oto Ami and Amitaro models, speedup of installation, etc.)
  - 2024-03-16: ver 2.4.1 (Change in installation method using bat file)
  - 2024-03-15: ver 2.4.0 (Large-scale refactoring and various improvements, libraryization)
  - 2024-02-26: ver 2.3 (Dictionary function and editor function)
  - 2024-02-09: ver 2.2
  - 2024-02-07: ver 2.1
  - 2024-02-03: ver 2.0 (JP-Extra)
  - 2024-01-09: ver 1.3
  - 2023-12-31: ver 1.2
  - 2023-12-29: ver 1.1
  - 2023-12-27: ver 1.0

This repository is based on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1 and Japanese-Extra, so many thanks to the original author!

**Overview**

- Based on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1 and Japanese-Extra, which generate emotionally rich voices based on the content of the input text, this project allows you to freely control the emotion and speech style with strength and weakness.
- Even if you don't know Git or Python (if you're a Windows user), you can easily install and train. You can also support training on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb)
- If you only want to use voice synthesis, you can run it on CPU without a GPU. If you only want to use voice synthesis, you can install it as a Python library with `pip install style-bert-vits2`. See [library.ipynb](/library.ipynb) for an example.
- It also includes an API server that can be used for collaboration with other services ([@darai0512](https://github.com/darai0512) PR, thank you).
- Originally, Bert-VITS2's strength is that it reads "fun-looking articles in a fun way, and sad-looking articles in a sad way", so even if the style is specified as the default, you can generate emotionally rich voices.

## Usage

- For how to use in CLI, please refer to [here](/docs/CLI.md).
- Please also refer to [Frequently Asked Questions](/docs/FAQ.md).

### Operating environment

It has been confirmed to work on Windows Command Prompt, WSL2, and Linux (Ubuntu Desktop) in each UI and API Server (please be careful with path specification in WSL). If you don't have an NVidia GPU, you can't train, but you can synthesize voices and merge them on CPU.

### Installation

For installation and usage examples as a Python library, please refer to [library.ipynb](/library.ipynb).

#### For those who are not familiar with Git or Python

Assuming Windows.

1. Download [this zip file](https://github.com/litagin02/Style-Bert-VITS2/releases/download/2.6.0/sbv2.zip) to a location **without Japanese or spaces in the path** and extract it.
  - If you have a GPU, double-click `Install-Style-Bert-VITS2.bat`.
  - If you don't have a GPU, double-click `Install-Style-Bert-VITS2-CPU.bat`. You can't train in the CPU version, but you can synthesize voices and merge them.
2. Wait for the necessary environment to be installed automatically.
3. After that, if the editor for automatic voice synthesis starts, the installation is successful. The default model is already downloaded, so you can play with it as it is.

If you want to update, double-click `Update-Style-Bert-VITS2.bat`.

However, if you want to update from version **2.4.1** or earlier, you need to delete everything and install it again. I'm sorry. Please refer to [CHANGELOG.md](/docs/CHANGELOG.md) for migration methods.

#### For those who can use Git or Python

Since [uv](https://github.com/astral-sh/uv), a Python virtual environment and package management tool, is faster than pip, I recommend using it for installation.
(If you don't want to use it, regular pip is fine.)

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
uv venv venv
venv\Scripts\activate
uv pip install "torch<2.4" "torchaudio<2.4" --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
python initialize.py  # Download necessary models and default TTS models
```
Don't forget the last part.

### Voice synthesis

The voice synthesis editor starts by double-clicking `Editor.bat` or running `python server_editor.py --inbrowser` (start in CPU mode with `--device cpu`). You can change the settings for each line in the screen, create a manuscript, and save, load, and edit the dictionary.
The default model is already downloaded at installation, so you can use it even if you haven't trained it.

The editor part is separated into a [separate repository](https://github.com/litagin02/Style-Bert-VITS2-Editor).

The voice synthesis WebUI before version 2.2 starts the voice synthesis tab by double-clicking `App.bat` or running `python app.py`. You can also open the voice synthesis tab by double-clicking `Inference.bat`.

The structure of the model files necessary for voice synthesis is as follows (you don't need to manually place them).
```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```
In this way, `config.json`, `*.safetensors`, and `style_vectors.npy` are required for inference. If you want to share a model, please share these three files.

Among these, `style_vectors.npy` is a file necessary to control the style, and the default average style "Neutral" is generated during training.
If you want to use multiple styles to control the style in more detail, please refer to "Style generation" below (even if you only use the average style, if the training data is emotionally rich, you can generate emotionally rich voices).

### Training

- For details on training in CLI, please refer to [here](docs/CLI.md). For details on training on paperspace, please refer to [here](docs/paperspace.md), and for training on colab, please refer to [here](http://colab.research.google.com/github/litagin02/Style-Bert-VITS2/blob/master/colab.ipynb).

To train, you need multiple audio files of 2-14 seconds and their transcription data.

- If you already have split audio files and transcription data from an existing corpus, you can use them as they are (modify the transcription file if necessary). Please refer to "Training WebUI" below.
- If you don't have that, you can use the tool included to create a dataset that can be used for training immediately from the audio files (the length doesn't matter).

#### Creating a dataset

- From the "Create dataset" tab of `App.bat` or `Dataset.bat`, you can slice the audio files to the appropriate length and then automatically transcribe the text. You can also open the dataset tab by double-clicking `Dataset.bat`.
- After following the instructions, you can train directly from the "Training" tab.

#### Training WebUI

- Please follow the instructions from the "Training" tab of the WebUI opened by double-clicking `App.bat` or running `python app.py`. You can also open the training tab by double-clicking `Train.bat`.

### Style generation

- By default, the default style "Neutral" and the style generated according to the folder separation of the training folder are generated.
- For those who want to manually create styles in other ways.
- From the "Style creation" tab of the WebUI opened by double-clicking `App.bat` or running `python app.py`, you can generate styles using audio files. You can also open the style vectors tab by double-clicking `StyleVectors.bat`.
- It is independent of training, so you can do it even during training, and you can do it over and over again after the training is finished (you need to finish the preprocessing).

### API Server

When you run `python server_fastapi.py` in the environment you built, the API server starts.
Please check the API specification after starting at `/docs`.

- The default maximum number of input characters is 100. This can be changed with `server.limit` in `config.yml`.
- By default, CORS settings allow all domains. As much as possible, please change the value of `server.origins` in `config.yml` and limit it to a trusted domain (you can disable CORS settings by deleting the key).

The API server for the voice synthesis editor starts with `python server_editor.py`. But it's not well maintained yet. Only the minimum necessary API is currently implemented from the [editor repository](https://github.com/litagin02/Style-Bert-VITS2-Editor).

Please refer to [this Dockerfile](Dockerfile.deploy) for the web deployment of the voice synthesis editor.

### Merge

You can mix two models in four points: "voice quality", "voice pitch", "emotional expression", and "tempo", and create a new model, or perform operations such as "add the difference of two other models to a certain model".
From the "Merge" tab of `App.bat` or `Merge.bat`, you can select two models and merge them. You can also open the merge tab by double-clicking `Merge.bat`.

### Naturalness evaluation

As a "single" indicator of which step numbers of the training results are good, we have prepared a script using [SpeechMOS](https://github.com/tarepan/SpeechMOS):
```bash
python speech_mos.py -m <model_name>
```
The naturalness evaluation for each step is displayed, and the results are saved in `mos_results` folder as `mos_{model_name}.csv` and `mos_{model_name}.png`. If you want to change the text you want to read, please adjust it yourself by manipulating the file. Also, since it is just an evaluation based on the standard of not considering accents, emotional expressions, and intonation at all, it is just one of the guidelines, so I think it is best to actually read and select.

## Bert-VITS2 and its relationship

Basically, it's just a slight modification of the model structure of Bert-VITS2. The [old pre-trained model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) and the [JP-Extra pre-trained model](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) are essentially the same as the original base model of Bert-VITS2 v2.1 and the [JP-Extra pre-trained model of Bert-VITS2](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra) (converted to safetensors by removing unnecessary weights).

Specifically, the following points are different.

- Like [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2), it's easy to use even for people who don't know Python or Git.
- Changed the model for emotion embedding (changed to 256-dimensional [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM), which is more for speaker identification than emotion embedding).
- Removed the vector quantization of emotion embedding and replaced it with a simple fully connected layer.
- By creating a style vector file `style_vectors.npy`, you can generate voices while specifying the strength of the effect continuously using that style.
- Created various WebUIs
- Support for training in bf16
- Support for safetensors format, and use safetensors by default
- Other minor bug fixes and refactoring

## References
In addition to the original reference (written below), I used the following repositories:
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)

[The pretrained model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) and [JP-Extra version](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) is essentially taken from [the original base model of Bert-VITS2 v2.1](https://huggingface.co/Garydesu/bert-vits2_base_model-2.1) and [JP-Extra pretrained model of Bert-VITS2](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra), so all the credits go to the original author ([Fish Audio](https://github.com/fishaudio)):


In addition, [text/user_dict/](text/user_dict) module is based on the following repositories:
- [voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)
and the license of this module is LGPL v3.

## LICENSE

This repository is licensed under the GNU Affero General Public License v3.0, the same as the original Bert-VITS2 repository. For more details, see [LICENSE](LICENSE).

In addition, [text/user_dict/](text/user_dict) module is licensed under the GNU Lesser General Public License v3.0, inherited from the original VOICEVOX engine repository. For more details, see [LGPL_LICENSE](LGPL_LICENSE).



Below is the original README.md.
---

<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# Bert-VITS2

VITS2 Backbone with multilingual bert

For quick guide, please refer to `webui_preprocess.py`.

简易教程请参见 `webui_preprocess.py`。

## 请注意，本项目核心思路来源于[anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS) 一个非常好的tts项目
## MassTTS的演示demo为[ai版峰哥锐评峰哥本人,并找回了在金三角失落的腰子](https://www.bilibili.com/video/BV1w24y1c7z9)

[//]: # (## 本项目与[PlayVoice/vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41; 没有任何关系)

[//]: # ()
[//]: # (本仓库来源于之前朋友分享了ai峰哥的视频，本人被其中的效果惊艳，在自己尝试MassTTS以后发现fs在音质方面与vits有一定差距，并且training的pipeline比vits更复杂，因此按照其思路将bert)

## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应当参阅代码自己学习如何训练。

### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途。
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
#### QQ Group：815818430
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
+ [emotional-vits](https://github.com/innnky/emotional-vits)
+ [fish-speech](https://github.com/fishaudio/fish-speech)
+ [Bert-VITS2-UI](https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI)
## 感谢所有贡献者作出的努力
<a href="https://github.com/fishaudio/Bert-VITS2/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/Bert-VITS2"/>
</a>

[//]: # (# 本项目所有代码引用均已写明，bert部分代码思路来源于[AI峰哥]&#40;https://www.bilibili.com/video/BV1w24y1c7z9&#41;，与[vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41;无任何关系。欢迎各位查阅代码。同时，我们也对该开发者的[碰瓷，乃至开盒开发者的行为]&#40;https://www.bilibili.com/read/cv27101514/&#41;表示强烈谴责。)
