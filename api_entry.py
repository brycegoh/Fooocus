import os
import sys
import time
import base64
import io

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


print("preparing environment...")
from launch import *

print("Starting inference worker")
from modules.async_worker import TaskParams, AsyncTask
from PIL import Image

params = TaskParams(
  prompt="Wooden cabinetry with black handles, white quartz countertop, pale green subway tile backsplash, built-in oven, stainless steel stove, glass-door upper cabinets with ceramic ware, hexagonal gray floor tiles, wooden table with matching stools, pendant lighting, large window with street view."
)

async_task = AsyncTask(params)

final_result = None

for result in async_task.start():
    if len(result) == 0:
        time.sleep(1)
        continue
    flag, product = result
    print(flag)
    if flag == 'preview':
        print(product)
    if flag == 'results':
        print(product)
    if flag == 'finish':
        final_result = product
        break

print(final_result)

def base64_to_pil(base64_string):
    decoded = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(decoded))

for idx, img_b64 in enumerate(final_result['images']):
    img = base64_to_pil(img_b64)
    img.save(f"output_{idx}.png", "PNG")