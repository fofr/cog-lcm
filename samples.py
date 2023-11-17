"""
A handy utility for verifying SDXL image generation locally.
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""

import base64
import os
import sys
import requests
import glob
import time

def gen(output_fn, **kwargs):
    if glob.glob(f"{output_fn}*"):
        return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    print(data)

    try:
        for i, datauri in enumerate(data["output"]):
            base64_encoded_data = datauri.split(",")[1]
            decoded_data = base64.b64decode(base64_encoded_data)
            with open(
                f"{output_fn.rsplit('.', 1)[0]}_{i}.{output_fn.rsplit('.', 1)[1]}", "wb"
            ) as f:
                f.write(decoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)


def main():
    total_time = 0
    for i in range(10):
        start_time = time.time()
        gen(
            f"sample_{i}.png",
            prompt="A studio portrait photo of a cat",
            seed=1000,
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        total_time += end_time - start_time
    average_time = total_time / 10
    print(f"Average time taken: {average_time} seconds")

if __name__ == "__main__":
    main()
