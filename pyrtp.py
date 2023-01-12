import asyncio
import websockets
import os
import torch
import socket
import json
import audioop
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
import random
from struct import pack, unpack
import time
import re
nltk.download('punkt')
print("Now ready to synthesize!")
# os.environ['CUDA_VISIBLE_DEVICES'] = '2',




# import IPython
# from IPython.display import display, Audio


trans_type = "phn"

dict_path = "/home/vectone/Desktop/tts/downloads/downloads-20230112T071813Z-001/downloads/en/tacotron2/data/lang_1phn/phn_train_no_dev_units.txt"

model_path = "/home/vectone/Desktop/tts/downloads/downloads-20230112T071813Z-001/downloads/en/tacotron2/exp-20230112T073544Z-001/exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best"

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

def Int2Float(sound):
    """converts the format and normalizes the data"""
    _sound = np.copy(sound) #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
       _sound *= 1/abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32


def fun(dataa):
    while True:
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
        '=!': 'not equals',
        "doesn't" :'doesnt',
        "couldn't":'couldnt',
        "wouldn't":'couldnt',
        "can't":'cant',
        "top-up":'top up',
        "OTP":'O T P',
        "ID":"I D",
        "We've":"We have",
        "447550640382":"4 4 7 5 5 0 6 4 0 3 8 2 ",
        "447589771910":"4 4 7 5 8 9 7 7 1 9 1 0",
        "447589412159":"4 4 7 5 8 9 4 1 2 1 5 9"

        }

        st=dataa   
        for key in dic.keys():
            st = st.replace(key, dic[key])
        
        with torch.no_grad():
            x = frontend(st)
            c, _, _ = model.inference(x, inference_args)
            y = vocoder.inference(c)         
        array = y.view(-1).cpu().numpy()
        array1 = Int2Float(array)
        #byte = array1.tobytes()       
        byt = float_to_byte(array1)
        byt1 = byt[0:-200]
        return byt1
       
HEADER_SIZE = 12
class RtpPacket:	
    header = bytearray(HEADER_SIZE)
    timestamp = 0
    seqnum = 0
    
    def encode(self,version, padding, extension, cc,marker, pt, ssrc,payload):
        """Encode the RTP packet with header fields and payload."""
        a=441
        self.timestamp = self.timestamp + a
        self.seqnum = self.seqnum + 1 
        header = bytearray(HEADER_SIZE + 441)
        chunk_size = 441 
                #--------------
                # TO COMPLETE
                #--------------
                # Fill the header bytearray with RTP header fields
        header[0] = (header[0]| version << 6) & 0xC0; # 2 bits
        header[0] = (header[0]| padding << 5); # 1 bit
        header[0] = (header[0]| extension << 4); # 1 bit
        header[0] = (header[0]| (cc & 0x0F)); # 4 bits
        header[1] = (header[1]| marker << 7); # 1 bit
        header[1] = (header[1]| (pt & 0x7f)); # 7 bits 
        header[2] = (self.seqnum & 0xFF00) >> 8; # 16 bits total
        header[3] = (self.seqnum & 0xFF); # second 8
        header[4] = (self.timestamp >> 24); # 32 bit
        header[5] = (self.timestamp >> 16) & 0xFF
        header[6] = (self.timestamp >> 8) & 0xFF
        header[7] = (self.timestamp & 0xFF)
        header[8] = (ssrc >> 24); # 32 bit
        header[9] = (ssrc >> 16) & 0xFF
        header[10] = (ssrc >> 8) & 0xFF
        header[11] = ssrc & 0xFF
        header[12:]=payload[:]
        return header



# create handler for each connection
async def handler(websocket, path):
    #await websocket.keep_alive() 
    data = await websocket.recv()
    data1=json.loads(data)
    ip=data1.get("ip") 
    port=data1.get("port") 
    print(ip,port)
    await websocket.send("we are getting the ip") 
    #await websocket.accept()
    while True:
        text=[]
        dataa = await websocket.recv()
        print(dataa)
        a=json.loads(dataa)
        text.append(a.get("text)"))        
        m = {"status": "Inprogress",
                    "error_code": 0,
                    "error_message": "Success"}
        await websocket.send(json.dumps(m)) 
        serverAddressPort = (ip, port)
        start = time.time()
        y=a.get("text")
        #if len(y)>25:
        j = re.split(r"[^a-zA-Z0-9\s?!-'.]", y)
        #j = re.split(r'[^a-zA-Z0-9\s]', y)
        for g in j:
            packet_bytes = fun(g)
            #print("the lenght of packet_bytes:",len(packet_bytes))
            local_IP="192.168.13.220"
            localport=8000
            UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            UDPServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            UDPServerSocket.bind((local_IP,localport))
            mv = memoryview(packet_bytes).cast('H') 
            chunk_size = 441          
            pcma_bytes = audioop.lin2alaw(packet_bytes,2)
            temb=0
            for i in range(441,len(pcma_bytes),chunk_size):                
                a=pcma_bytes[temb:i]
                temb=i
                rtpPkt = RtpPacket.encode(RtpPacket,2,0,0,0,0,8,1,a)
                UDPServerSocket.sendto(rtpPkt,serverAddressPort)
                start_time = time.time()                   
                elapsed_time = time.time() - start_time
                remaining_time=0.02 - elapsed_time
                time.sleep(remaining_time)
               
            b=pcma_bytes[temb:]
            data_need=441-len(b)
            data=b'\x00'*data_need
            c=b+data
            rtpPkt = RtpPacket.encode(RtpPacket,2,0,0,0,0,8,1,c)
            UDPServerSocket.sendto(rtpPkt,serverAddressPort) 
        s=time.time()-start
        print("the total time:",s)             
        print(serverAddressPort)
            
        print("data sent")
        
        k = {"status": "Completed",
                            "error_code": 0,
                            "error_message": "Success"}     
        await websocket.send(json.dumps(k))
      
        


start_server = websockets.serve(handler, "192.168.13.220", 8000,ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever() 
