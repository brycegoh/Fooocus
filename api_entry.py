import os
import sys
import time


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


print("preparing environment...")
from launch import *

print("Starting inference worker")
from modules.async_worker import TaskParams, AsyncTask

params = TaskParams(
  prompt="Wooden cabinetry with black handles, white quartz countertop, pale green subway tile backsplash, built-in oven, stainless steel stove, glass-door upper cabinets with ceramic ware, hexagonal gray floor tiles, wooden table with matching stools, pendant lighting, large window with street view."
)

async_task = AsyncTask(params)

for result in async_task.start():
    if len(result) == 0:
        time.sleep(1)
        continue
    print(result)
    
    if result[0] == 'finish':
        break