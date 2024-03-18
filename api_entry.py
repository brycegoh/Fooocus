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
from modules.segmentationMaskGenerator import SegmentationMaskGenerator
from modules.util import base64_to_np, base64_to_pil, pil_to_base64
from modules.flags import ip_list
from PIL import Image
import numpy as np

input_params = {
    'prompt': "Wooden cabinetry with black handles, white quartz countertop, pale green subway tile backsplash, built-in oven, stainless steel stove, glass-door upper cabinets with ceramic ware, hexagonal gray floor tiles, wooden table with matching stools, pendant lighting, large window with street view.",
    'guidance_scale': 4.0,
    'image_prompts': [
        {
            'image': pil_to_base64(Image.open("./kitchen.jpg")),
            'stop': 0.5,
            'weight': 0.6,
            'type': "Depth",
        },
        {
            'image': pil_to_base64(Image.open("./kitchen.jpg")),
            'stop': 0.5,
            'weight': 0.6,
            'type': "PyraCanny",
        }
    ],
    'inpaint_input_image': {
        'image': pil_to_base64(Image.open("./kitchen.jpg")),
        'classes_to_avoid': ['wall'],
    },
}



task_params = TaskParams()
task_params.map_dict_to_self(input_params)

if 'inpaint_input_image' in input_params and input_params['inpaint_input_image'].get('image', None) is not None:
    task_params.inpaint_input_image = {}
    classes_to_avoid = input_params['inpaint_input_image'].get('classes_to_avoid', ['wall'])
    inpaint_img = input_params['inpaint_input_image'].get('image')
    inpaint_img = base64_to_pil(inpaint_img)
    segmentation_mask_generator = SegmentationMaskGenerator()
    mask = segmentation_mask_generator.get_mask(inpaint_img, classes_to_avoid)
    task_params.inpaint_input_image['image'] = np.array(inpaint_img)
    task_params.inpaint_input_image['mask'] = mask
    Image.fromarray(mask).save("mask.png", "PNG")

if 'image_prompts' in input_params:
    task_params.image_prompts = []
    for img_prompt in input_params['image_prompts']:
        img = img_prompt.get('image', None)
        stop = img_prompt.get('stop', None)
        weight = img_prompt.get('weight', None)
        type = img_prompt.get('type', None)
        if img is None or stop is None or weight is None or type is None:
            raise ValueError("Invalid image prompt")
        if type not in ip_list:
            raise ValueError("Invalid image prompt type")
        task_params.image_prompts.append([
            base64_to_np(img),
            stop,
            weight,
            type
        ])

async_task = AsyncTask(task_params)

final_result = None
for result in async_task.start():
    if len(result) == 0:
        time.sleep(1)
        continue
    flag, product = result
    print(flag)
    if flag == 'preview':
        continue
    if flag == 'results':
        print(product)
    if flag == 'finish':
        final_result = product
        break

for idx, img_b64 in enumerate(final_result):
    img = base64_to_pil(img_b64)
    img.save(f"output_{idx}.png", "PNG")