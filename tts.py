import asyncio
import websockets
import os
import re
import torch
import socket
import json
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
import audioop
import time
import kaitaistruct
from kaitaistruct import KaitaiStream, BytesIO
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from enum import Enum
import random
# import websockets
# import asyncio
# import socket
# import json
nltk.download('punkt')
print("Now ready to synthesize!")
# os.environ['CUDA_VISIBLE_DEVICES'] = '2',




# import IPython
# from IPython.display import display, Audio


trans_type = "phn"

dict_path = "/home/vectone/Desktop/tts/downloads/en/tacotron2/data/lang_1phn/phn_train_no_dev_units.txt"

model_path = "/home/vectone/Desktop/tts/downloads/en/tacotron2/exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best"

vocoder_path = "/home/vectone/Desktop/tts/downloads/en/parallel_wavegan/ljspeech.parallel_wavegan.v2/checkpoint-400000steps.pkl"


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
        # print(f"Cleaned text: {text}")
        charseq = text.split(" ")
    else:
        # print(f"Cleaned text: {text}")
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


def GenerateRTP(packet_vars):

    #The first twelve octates are present in every RTP packet.
    #The first octet is the version number of the RTP packet.
    #The second octet is the padding bit.
    #The third octet is the extension bit.
    #The fourth octet is the CSRC count.
    #The fifth octet is the marker bit.
    #The sixth octet is the payload type.
    #The seventh to twelve octets are the sequence number.
    #The thirteen to eighteen octets are the timestamp.
    #The nineteen to twenty-four octets are the synchronization source (SSRC).
    #The remaining octets are the payload data.

    #Generate fist byte of the header a binary string:
    version = format(packet_vars['version'], 'b').zfill(2)
    padding = format(packet_vars['padding'], 'b')
    extension = format(packet_vars['extension'], 'b')
    csrc_count = format(packet_vars['csi_count'], 'b').zfill(4)
    
    byte1 = format(int((version + padding + extension + csrc_count), 2), 'x').zfill(2)


    #Generate second byte of the header as binary string:
    marker = format(packet_vars['marker'], 'b')
    payload_type = format(packet_vars['payload_type'], 'b').zfill(7)

    byte2 = format(int((marker + payload_type), 2), 'x').zfill(2)

    sequence_number = format(packet_vars['sequence_number'], 'x').zfill(4)
    timestamp = format(packet_vars['timestamp'], 'x').zfill(8)
    ssrc = format(packet_vars['ssrc'], 'x').zfill(8)

    payload = packet_vars['payload']

    packet = byte1 + byte2 + sequence_number + timestamp + ssrc + payload

    return packet.encode()
def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size][0]




def fun(dataa):
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

    st=dataa   
    for key in dic.keys():
        st = st.replace(key, dic[key])
    with torch.no_grad():
        x = frontend(st)
        c, _, _ = model.inference(x, inference_args)
        y = vocoder.inference(c)         
        array = y.view(-1).cpu().numpy()        
        byte = array.tobytes()       
        byt = float_to_byte(array)
        
        # print("The byt size is :",sys.getsizeof(byt))
        mv = memoryview(byt).cast('H') 
        pcma_bytes = audioop.lin2alaw(mv,1)
        chunk_size = 441        
        # ff = list(split(byt, chunk_size))
        for i in range(0, len(pcma_bytes), chunk_size):            
            pcm_bytes = audioop.alaw2lin(pcma_bytes[i], 2) 
            packet_vars = {'version' : 2,
                        'padding' : 0,
                        'extension' : 0,
                        'csi_count' : 0,
                        'marker' : 0,
                        'payload_type' : 97,
                        'sequence_number' : random.randint(1,61),
                        'timestamp' : random.randint(1,9999),
                        'ssrc' : 185755418,
                        'payload' : pcm_bytes}
            rtp_packet = GenerateRTP(packet_vars)        
            return rtp_packet         
      
            










# create handler for each connection
async def handler(websocket, path): 
    data = await websocket.recv()
    data1=json.loads(data)
    IP=data1.get("ip") 
    PORT=data1.get("port") 
    await websocket.send("we are getting the ip") 
    while True:
        text=[]
        dataa = await websocket.recv()
        # print(dataa)
        a=json.loads(dataa)
        b=a.get("text")
        c=re.split("",b)
        packet_bytes = fun(a.get("text"))  
        
        
        
        text.append(a.get("text)"))
        m = {"status": "Inprogress",
                    "error_code": 0,
                    "error_message": "Success"}
        await websocket.send(json.dumps(m)) 
        serverAddressPort = (IP, PORT)
        packet_bytes = fun(a.get("text"))                      
        k = {"status": "Completed",
                             "error_code": 0,
                            "error_message": "Success"}
        
        UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)        
        UDPServerSocket.sendto(packet_bytes,serverAddressPort)
        await websocket.send(json.dumps(k))
        print("the data is sent")    



start_server = websockets.serve(handler, "192.168.13.220", 8001)

asyncio.get_event_loop().run_until_complete(start_server)

asyncio.get_event_loop().run_forever() 
      
           

