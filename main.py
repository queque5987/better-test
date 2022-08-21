
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

# import rtvc_main
# from encoder import inference as encoder
# from synthesizer.inference import Synthesizer

# args = rtvc_main.rtvc_args()
# synthesizer = Synthesizer(args.syn_model_fpath)
# encoder.load_model(args.enc_model_fpath)

app = FastAPI()

class UserInput(BaseModel):
    user_input: float

@app.get('/')
async def index():
    return {"Message": "This is Index"}

# @app.post('/predict/')
# async def predict(UserInput: UserInput):

#     preprocessed_wav = encoder.preprocess_wav([UserInput.user_input])
#     embed = encoder.embed_utterance(preprocessed_wav)

#     return {"embed": float(embed)}