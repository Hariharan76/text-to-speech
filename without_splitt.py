import asyncio
import websockets
import os
import torch
import sys
import json

from argparse import Namespace
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
# define neural vocoder
from parallel_wavegan.utils import load_model
# define text frontend
from tacotron_cleaner.cleaners import custom_english_cleaners
from g2p_en import G2p
import nltk
import numpy as np
import time


# import IPython
# from IPython.display import display, Audio


trans_type = "phn"

dict_path = "/home/vectone/PycharmProjects/pythonProject/pythonProject/tacotrom-tts/downloads/en/tacotron2/data/lang_1phn/phn_train_no_dev_units.txt"

model_path = "/home/vectone/PycharmProjects/pythonProject/pythonProject/tacotrom-tts/downloads/en/tacotron2/exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best"

vocoder_path = "/home/vectone/PycharmProjects/pythonProject/pythonProject/tacotrom-tts/downloads/en/parallel_wavegan/ljspeech.parallel_wavegan.v2/checkpoint-400000steps.pkl"


# pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model_path)
# pruned_model.summary()
sys.path.append("espnet")

device = torch.device("cuda")

idim, odim, train_args = get_model_conf(model_path)
model_class = dynamic_import(train_args.model_module)
model = model_class(idim, odim, train_args)
torch_load(model_path, model)
model = model.eval().to(device)
inference_args = Namespace(**{
    "threshold": 0.5, "minlenratio": 0.0, "maxlenratio": 10.0,
    # Only for Tacotron 2
    "use_attention_constraint": True, "backward_window": 1, "forward_window": 3,
    # Only for fastspeech (lower than 1.0 is faster speech, higher than 1.0 is slower speech)
    "fastspeech_alpha": 1.0,
})
dic = {
    '@': ' at ',
    '#': 'number',
    '$': 'dollar',
    '.com': ' dot com',
    '&': 'And',
    '%': 'percentage',
    '+': 'plus',
    '=': 'equals',
    '<': 'less than',
    '>': 'greater than',
    '>=': 'greater than or equals',
    '<=': 'less than or equals',
    '=!': 'not equals'

}



fs = 22050
vocoder = load_model(vocoder_path)
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)



with open(dict_path) as f:
    lines = f.readlines()
lines = [line.replace("\n", "").split(" ") for line in lines]
char_to_id = {c: int(i) for c, i in lines}
g2p = G2p()


def frontend(text):
    """Clean text and then convert to id sequence."""
    text = custom_english_cleaners(text)

    if trans_type == "phn":
        text = filter(lambda s: s != " ", g2p(text))
        text = " ".join(text)
        print(f"Cleaned text: {text}")
        charseq = text.split(" ")
    else:
        print(f"Cleaned text: {text}")
        charseq = list(text)
    idseq = []
    for c in charseq:
        if c.isspace():
            idseq += [char_to_id["<space>"]]
        elif c not in char_to_id.keys():
            idseq += [char_to_id["<unk>"]]
        else:
            idseq += [char_to_id[c]]
    idseq += [idim - 1]  # <eos>
    return torch.LongTensor(idseq).view(-1).to(device)


"""Helper functions for working with audio files in NumPy."""
"""some code borrowed from https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py"""

import numpy as np
import contextlib
import librosa
import struct
import soundfile

def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return float2pcm(sig, dtype='int16').tobytes()

def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)



@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for temporarily setting NumPy print options.
    See http://stackoverflow.com/a/2891805/500098
    """
    original = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwargs)
        yield
    finally:
        np.set_printoptions(**original)



nltk.download('punkt')
print("Now ready to synthesize!")
#print("Audio started playing")

# create handler for each connection

async def handler(websocket, path):


    while True:

        b = await websocket.recv()
        z=json.loads(b)
        m = {"status": "Inprogress",
                    "error_code": 0,
                    "error_message": "Success"}
        n = json.dumps(m)
        print(n)
        await websocket.send(n)
        st = z["text"]
        
        start = time.time()
        print(len(st))
        print(b)
        for key in dic.keys():
            st = st.replace(key, dic[key])
       
        with torch.no_grad():
            x = frontend(st)
            c, _, _ = model.inference(x, inference_args)
            y = vocoder.inference(c)
            
        
        array = y.view(-1).cpu().numpy()
        
        byte = array.tobytes()
        
        byt = float_to_byte(array)
       
        print("The byt size is :",sys.getsizeof(byt))
        
        # with open("result.pcm",'wb') as f:
        #     f.write(byt)
        time_taken = (time.time() - start)
        print("The time taken is :",time_taken)
        # cache = {}

        # def get_audio(byt):
        #     if byt in cache:
        #         return cache[byt]
        #     else:
        #         pass
                
        await websocket.send(byt)

        key = {"status": "Completed",
                             "error_code": 0,
                            "error_message": "Success"}
        p=json.dumps(key)
        print(p) 
        
        # await websocket.send(byt)
        # await websocket.send(n)
        await websocket.send(p)
        continue
           
   
start_server = websockets.serve(handler, "0.0.0.0", 8000)

asyncio.get_event_loop().run_until_complete(start_server)

asyncio.get_event_loop().run_forever()
