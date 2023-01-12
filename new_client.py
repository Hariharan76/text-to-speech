import asyncio
import websockets
import json
import socket
 
async def test():
    async with websockets.connect('ws://192.168.13.220:8001/') as websocket:
        a={"IP":'127.0.0.1',"port":20001}
        await websocket.send(json.dumps(a))
        response = await websocket.recv()
        print(response)
        while True:
            data=input("enter the input:")
            dataa={"text":data}
            await websocket.send(json.dumps(dataa))
            t=await websocket.recv()
            print(t)
            ta=await websocket.recv()
            print(ta)
            UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)            
            localIP     = "127.0.0.1"
            localPort   = 20001
            bufferSize  = 1024
            UDPServerSocket.bind((localIP, localPort))
            # bufferSize          = 1024
            # UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            msgFromServer,addr = UDPServerSocket.recvfrom(bufferSize)
            print(msgFromServer,addr)
            
 
asyncio.get_event_loop().run_until_complete(test())