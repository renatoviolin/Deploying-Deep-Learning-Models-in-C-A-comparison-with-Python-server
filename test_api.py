# %% ---------------------------------------------
import requests
import json
import base64
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import time
import numpy as np


# %% ---------------------------------------------
n_request = 100
URL = "http://localhost:8000/predict"
image_path = "./img/cat.jpg"
byte_content = base64.b64encode(open(image_path, "rb").read())
img = byte_content.decode('utf-8')


# %% ---------------------------------------------
async def _async_fetch(session, data):
    async with session.post(URL, json=data) as response:
        r = await response.text()
        return r


async def run():
    async with aiohttp.ClientSession() as session:
        tasks = [_async_fetch(session, data={"image": img, "request_id": i}) for i in range(n_request)]
        results = await asyncio.gather(*tasks)
    return results


# %% ---------------------------------------------
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    times = []
    print(f'Starting {n_request} request asynchronous')
    for i in range(1, 11, 1):
        start = time.perf_counter()
        loop.run_until_complete(run())
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f'Run {i}: {elapsed:.5f} seconds')

    loop.close()
    print(f'Mean: {np.mean(times):.5f} seconds')
