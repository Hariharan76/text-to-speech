import os
import asyncio
import websockets
import random

websocket_clients = set()

async def handle_socket_connection(websocket, path):
    """Handles the whole lifecycle of each client's websocket connection."""
    websocket_clients.add(websocket)
    print(f'New connection from: {websocket.remote_address} ({len(websocket_clients)} total)')
    try:
        # This loop will keep listening on the socket until its closed.
        async for raw_message in websocket:
            print(f'Got: [{raw_message}] from socket [{id(websocket)}]')
    except websockets.exceptions.ConnectionClosedError as cce:
        pass
    finally:
        print(f'Disconnected from socket [{id(websocket)}]...')
        websocket_clients.remove(websocket)

async def broadcast_random_number(loop):
    """Keeps sending a random # to each connected websocket client"""
    while True:
        for c in websocket_clients:
            num = str(random.randint(10, 99))
            print(f'Sending [{num}] to socket [{id(c)}]')
            await c.send(num)
        await asyncio.sleep(2)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        socket_server = websockets.serve(handle_socket_connection, 'localhost', 6789)
        print(f'Started socket server: {socket_server} ...')
        loop.run_until_complete(socket_server)
        loop.run_until_complete(broadcast_random_number(loop))
        loop.run_forever()
    finally:
        loop.close()
        print(f"Successfully shutdown [{loop}].")