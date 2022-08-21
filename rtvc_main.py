import os
from tabnanny import check
import torch
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

import collections

class rtvc_args():
    def __init__(self):
        self.enc_model_fpath = Path("saved_models/default/encoder.pt")
        self.syn_model_fpath = Path("saved_models/default/synthesizer.pt")
        self.voc_model_fpath = Path("saved_models/default/vocoder.pt")
        self.cpu = True
        self.seed = None
    def pop(self, idx):
        if idx == "cpu":
            return self.cpu


def from_embed(embed, synthesizer):
    text = "This project gives opportunity to the deaf"
    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")
    ## Generating the waveform
    print("Synthesizing the waveform:")
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)
    jack = generated_wav.tolist()
    generated_wav = np.array(jack)
    filename = "demo_output_fromembeds_%02d.wav" % 181817
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    return jack


def inference(original_wav, sampling_rate, text, args = rtvc_args()):
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)

    # in_fpath = Path("26-495-0000.wav")
    # preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # # - If the wav is already loaded:
    # original_wav, sampling_rate = librosa.load(str(in_fpath))
    # print(type(original_wav))
    original_wav = np.array(original_wav)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("Loaded file succesfully")
    embed = encoder.embed_utterance(preprocessed_wav)
    print(type(embed))
    print("Created the embedding")
    # text = "This project gives opportunity to the deaf"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")
    ## Generating the waveform
    print("Synthesizing the waveform:")

    # If seed is specified, reset torch seed and reload vocoder
    if args.seed is not None:
        torch.manual_seed(args.seed)
        vocoder.load_model(args.voc_model_fpath)
    
    generated_wav = vocoder.infer_waveform(spec)
    # generated_wav = synthesizer.griffin_lim(spec) #좃구림


    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)
    # filename = "demo_output_%02d.wav" % 2829
    return generated_wav.astype(np.float32)
    # print(generated_wav.dtype)
    # print(type(generated_wav))
    # sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    # print("\nSaved output as %s\n\n" % filename)

if __name__ == "__main__":

    args = rtvc_args()

    checkpoint = torch.load(args.voc_model_fpath, 'cpu')
    tentensor = checkpoint["model_state"]
    print(type(tentensor))
    tenlist = list(tentensor.items())
    print(len(tenlist))
    print(tenlist[3][1:][0].size())
    path = "saved_models/vocoder/"
    temp_path = path
    tensors = []
    for i, ten in enumerate(tenlist):
        tensor = ten[1:][0]
        # tensorr = ten[1]
        # print(tensor == tensorr)
        torch.save(tensor, os.path.join(path, "{}_{}.pt".format(str(i).zfill(3), ten[0])))
    # checkpoints = collections.OrderedDict()
    # for pt in os.listdir(path):
    #     tensr = torch.load(os.path.join(path, pt))
    #     checkpoints[pt[4:-3]] = tensr
    # # print(tentensor==checkpoints)
    # synthesizer = Synthesizer(args.syn_model_fpath)
    # synthesizer.model_state_load(checkpoints)
    # synthesizer.load()

    # # if args.pop("cpu"):
    # #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # # if torch.cuda.is_available():
    # #     device_id = torch.cuda.current_device()
    # #     gpu_properties = torch.cuda.get_device_properties(device_id)
    # #     ## Print some environment information (for debugging purposes)
    # #     print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
    # #         "%.1fGb total memory.\n" %
    # #         (torch.cuda.device_count(),
    # #         device_id,
    # #         gpu_properties.name,
    # #         gpu_properties.major,
    # #         gpu_properties.minor,
    # #         gpu_properties.total_memory / 1e9))
    # # else:
    # #     print("Using CPU for inference.\n")

    # # ensure_default_models(Path("saved_models"))
    # encoder.load_model(args.enc_model_fpath)
    # synthesizer = Synthesizer(args.syn_model_fpath)
    # vocoder.load_model(args.voc_model_fpath)

    # in_fpath = Path("26-495-0000.wav")
    # # preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # # # - If the wav is already loaded:
    # original_wav, sampling_rate = librosa.load(str(in_fpath))

    # text = "This is the prototype of the better project."

    # wavwav = inference(original_wav.tolist(), sampling_rate, text, args)
    # sf.write("filename.wav", wavwav, synthesizer.sample_rate)

    # # print(type(original_wav))
    # # asf = original_wav.tolist()
    # # original_wav = np.array(asf)
    # # preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    # # print("Loaded file succesfully")
    # # embed = encoder.embed_utterance(preprocessed_wav)
    # # print(type(embed))
    # # print("Created the embedding")
    # # text = "This project gives opportunity to the deaf"
    # # if args.seed is not None:
    # #     torch.manual_seed(args.seed)
    # #     synthesizer = Synthesizer(args.syn_model_fpath)

    # # # The synthesizer works in batch, so you need to put your data in a list or numpy array
    # # texts = [text]
    # # embeds = [embed]
    # # # If you know what the attention layer alignments are, you can retrieve them here by
    # # # passing return_alignments=True
    # # specs = synthesizer.synthesize_spectrograms(texts, embeds)
    # # spec = specs[0]
    # # print("Created the mel spectrogram")
    # # ## Generating the waveform
    # # print("Synthesizing the waveform:")

    # # # If seed is specified, reset torch seed and reload vocoder
    # # if args.seed is not None:
    # #     torch.manual_seed(args.seed)
    # #     vocoder.load_model(args.voc_model_fpath)

    # # """
    # # waveglow testing
    # # """
    # # # waveglow.write_wav(spec, "powered spec.wav")

    # # # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # # # spectrogram, the more time-efficient the vocoder.
    
    # # generated_wav = vocoder.infer_waveform(spec)
    # # # generated_wav = synthesizer.griffin_lim(spec) #좃구림


    # # ## Post-generation
    # # # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # # # pad it.
    # # generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # # # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    # # generated_wav = encoder.preprocess_wav(generated_wav)
    # # asf = generated_wav.tolist()
    # # generated_wav = np.array(asf)
    # # filename = "demo_output_%02d.wav" % 2829
    # # print(generated_wav.dtype)
    # # print(type(generated_wav))
    # # sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    # # print("\nSaved output as %s\n\n" % filename)