from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import os

import numpy as np

import rtvc_main
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

import soundfile as sf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = FastAPI()

class EmbedInput(BaseModel):
    wav: list
    sr: int
    text: str

@app.get('/')
def index():
    return {"Message": "This is Index"}

@app.get('/sibal')
def index(userinput: EmbedInput):
    u = userinput.dict()
    u = np.array(u['wav'])
    return u.tolist()

@app.get('/encoder/inference/')
def inference(userinput: EmbedInput):
    args = rtvc_main.rtvc_args()
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

    userinput = userinput.dict()
    sr = userinput["sr"]
    synthesizer.sample_rate = sr
    wav = userinput["wav"]
    wav = np.array(wav)
    text = userinput["text"]

    preprocessed_wav = encoder.preprocess_wav(wav, synthesizer.sample_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    # jack = rtvc_main.from_embed(embed, synthesizer)
    texts = [text]
    embeds = [embed]
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)
    # generated_wav = generated_wav.astype(np.float32)
    # return generated_wav.tolist()
    gen_wav = jsonable_encoder(generated_wav.tolist())
    gen_wav.append(16000)
    return JSONResponse(gen_wav)
