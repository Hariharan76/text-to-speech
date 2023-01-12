import numpy as np
import audioop
import sys
ds = "Hi! Hello! how do you do? Hi! Hello! how do you do? I am feeling great Hi! Hello! how do you do? I am feeling greatI am feeling great Hi! Hello! how do you do? I am feeling great Hi! Hello! how do you do? I am feeling great "
pcm_array = np.array(ds)
print(type(pcm_array))
# pcm_array = ds.split()
print(pcm_array)
pcm_array1= sys.getsizeof(pcm_array)

    
# pcm_array2 = np.array_split(pcm_array1, 3)
# Assume that `pcm_array` is a NumPy array containing PCM audio data
pcm_bytes = pcm_array.tobytes()
mv = memoryview(pcm_bytes).cast('H')

def split(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
      return i
    

chunk_size = 441
#my_list = [1,2,3,4,5,6,7,8,9]
ff = list(split(pcm_bytes, chunk_size))
pcma_bytes = audioop.lin2alaw(mv,1)

# Assume that `pcma_bytes` is a bytes object containing PCMA audio data
pcm_bytes = audioop.alaw2lin(pcma_bytes, 2)

# # Convert the bytes object back to a NumPy array
pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)


