
import asyncio
import json
import websockets
#from IPython.display import display, Audio
import time


async def test():
    async with websockets.connect('ws://localhost:8000') as websocket:
        #x = {"text": " "}
        # a=json.loads(x)
        #b = x.get('text')
        #print(type(x))
        print("Input your favorite sentence in English!")
        a = []
        
        while True:
            b = input()
            y={"text":b}
            start = time.time()
            await websocket.send(json.dumps(y))
            #respons2 = await websocket.recv()
            respons1 = await websocket.recv()
            #end = time.time()
            #respons3 = await websocket.recv() 
            #print(respons2)           
            print(respons1)
           
            #print(respons3)
            #print(end - start)
            # print(respons2)s
            # print(respons3)

asyncio.get_event_loop().run_until_complete(test())


