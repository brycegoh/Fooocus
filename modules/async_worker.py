import threading
from modules.patch import PatchSettings, patch_settings, patch_all
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
import random
from modules import config


patch_all()

class TaskParams(object):
    def __init__(
        self, 
        prompt: str = '',
        uov_input_image: np.ndarray | None = None,
        uov_method: str = 'Disabled', # [ 'Disabled', 'Vary (Subtle)', 'Vary (Strong)', 'Upscale (1.5x)', 'Upscale (2x)', 'Upscale (Fast 2x)', 'Upscale (Custom)']
        outpaint_selections: List[str] = [], # [ 'top', 'bottom', 'left', 'right' ]
        image_seed: int | None = None,
        sharpness: float = config.default_sample_sharpness,
        output_format: str = config.default_output_format,
        performance_selection: str = config.default_performance, # 'Quality', 'Speed', 'Extreme Speed'
        inpaint_input_image: Dict[str, np.ndarray] | None = None, # { 'mask': np.ndarray, 'image': np.ndarray  }
        image_prompts: List[Tuple[np.ndarray, float, float, str]] = [], # cn_img, cn_stop, cn_weight, cn_type (ImagePrompt, FaceSwap, PyraCanny, CPDS, Depth)
        inpaint_additional_prompt: str = '',
        guidance_scale: float =  config.default_cfg_scale,
        negative_prompt: str = '',
        aspect_ratios_selection: str = config.default_aspect_ratio,
        image_number: int = config.default_image_number,
        base_model_name: str = config.default_base_model_name,
        refiner_model_name: str = config.default_refiner_model_name,
        refiner_switch: float =  config.default_refiner_switch,
        loras: List[Tuple[str, float]] = config.default_loras, #[ [lora_model, lora_weight] ]
        style_selections: List[str] = config.default_styles,
        # advanced params
        disable_preview: bool = True,
        adm_scaler_positive: float = 1.5,
        adm_scaler_negative: float = 0.8,
        adm_scaler_end: float = 0.3,
        adaptive_cfg: float = 7.0,
        sampler_name: str = config.default_sampler,
        scheduler_name: str = config.default_scheduler,
        generate_image_grid: bool = False,
        overwrite_step: int = -1,
        overwrite_switch: int = -1,
        overwrite_width: int = -1,
        overwrite_height: int = -1,
        overwrite_vary_strength: float = -1,
        overwrite_upscale_strength: float = -1,
        mixing_image_prompt_and_vary_upscale: bool = False,
        mixing_image_prompt_and_inpaint: bool = False,
        debugging_cn_preprocessor: bool = False,
        skipping_cn_preprocessor: bool = False,
        controlnet_softness: float = 0.25,
        canny_low_threshold: int = 64,
        canny_high_threshold: int = 128,
        refiner_swap_method: str = 'joint',
        freeu_enabled: bool = False,
        freeu_b1: float | None = None,
        freeu_b2: float | None = None,
        freeu_s1: float | None = None,
        freeu_s2: float | None = None,
        debugging_inpaint_preprocessor: bool = False,
        inpaint_disable_initial_latent: bool = False,
        inpaint_engine: str =  'v2.6',
        inpaint_strength: float = 1.0,
        inpaint_respective_field: float = 0.618,
        inpaint_mask_upload_checkbox: bool = False,
        invert_mask_checkbox: bool = False,
        inpaint_erode_or_dilate: int = 0
    ):
        self.prompt = prompt
        self.performance_selection = performance_selection
        self.image_seed = image_seed
        self.sharpness = sharpness
        self.uov_input_image = uov_input_image
        self.uov_method = uov_method
        self.outpaint_selections = outpaint_selections
        self.output_format = output_format
        self.inpaint_input_image = inpaint_input_image
        self.image_prompts = image_prompts
        self.inpaint_additional_prompt = inpaint_additional_prompt
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.aspect_ratios_selection = aspect_ratios_selection
        self.image_number = image_number
        self.base_model_name = base_model_name
        self.refiner_model_name = refiner_model_name
        self.refiner_switch = refiner_switch
        self.loras = loras
        self.style_selections = style_selections
        self.disable_preview = disable_preview
        self.adm_scaler_positive = adm_scaler_positive
        self.adm_scaler_negative = adm_scaler_negative
        self.adm_scaler_end = adm_scaler_end
        self.adaptive_cfg = adaptive_cfg
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.generate_image_grid = generate_image_grid
        self.overwrite_step = overwrite_step
        self.overwrite_switch = overwrite_switch
        self.overwrite_width = overwrite_width
        self.overwrite_height = overwrite_height
        self.overwrite_vary_strength = overwrite_vary_strength
        self.overwrite_upscale_strength = overwrite_upscale_strength
        self.mixing_image_prompt_and_vary_upscale = mixing_image_prompt_and_vary_upscale
        self.mixing_image_prompt_and_inpaint = mixing_image_prompt_and_inpaint
        self.debugging_cn_preprocessor = debugging_cn_preprocessor
        self.skipping_cn_preprocessor = skipping_cn_preprocessor
        self.controlnet_softness = controlnet_softness
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.refiner_swap_method = refiner_swap_method
        self.freeu_enabled = freeu_enabled
        self.freeu_b1 = freeu_b1
        self.freeu_b2 = freeu_b2
        self.freeu_s1 = freeu_s1
        self.freeu_s2 = freeu_s2
        self.debugging_inpaint_preprocessor = debugging_inpaint_preprocessor
        self.inpaint_disable_initial_latent = inpaint_disable_initial_latent    
        self.inpaint_engine = inpaint_engine
        self.inpaint_strength = inpaint_strength
        self.inpaint_respective_field = inpaint_respective_field
        self.inpaint_mask_upload_checkbox = inpaint_mask_upload_checkbox
        self.invert_mask_checkbox = invert_mask_checkbox
        self.inpaint_erode_or_dilate = inpaint_erode_or_dilate
        
        if len(self.image_prompts) > 0 and inpaint_input_image is not None:
            self.mixing_image_prompt_and_inpaint = True
        if image_seed is None:
            MIN_SEED = 0
            MAX_SEED = 2**63 - 1
            self.image_seed = random.randint(MIN_SEED, MAX_SEED)

    def map_dict_to_self(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if len(self.image_prompts) > 0 and self.inpaint_input_image is not None:
            self.mixing_image_prompt_and_inpaint = True

class AsyncTask:
    def __init__(self, args: TaskParams):
        self.args = args
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False

    def start(self):
        global async_tasks
        import ldm_patched.modules.model_management as model_management
        import time

        with model_management.interrupt_processing_mutex:
            model_management.interrupt_processing = False
        execution_start_time = time.perf_counter()
        finished = False

        async_tasks.append(self)

        while not finished:
            time.sleep(0.01)
            if len(self.yields) > 0:
                r = self.yields.pop(0)
                flag, _ = r
                if flag == 'finish':
                    finished = True
                yield r

        execution_time = time.perf_counter() - execution_start_time
        print(f'Total time: {execution_time:.2f} seconds')
        return

async_tasks: List[AsyncTask] = []


def worker():
    global async_tasks

    import os
    import traceback
    import math
    import numpy as np
    import cv2
    import torch
    import time
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.patch
    import ldm_patched.modules.model_management
    import extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import extras.ip_adapter as ip_adapter
    import extras.face_crop
    import fooocus_version
    import args_manager

    from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion, apply_arrays
    from modules.private_logger import log
    from extras.expansion import safe_str
    from modules.util import remove_empty_str, HWC3, resize_image, \
        get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate, ordinal_suffix
    from modules.upscaler import perform_upscale
    from modules.flags import Performance
    from modules.meta_parser import get_metadata_parser, MetadataScheme

    pid = os.getpid()
    print(f'Started worker with PID {pid}')

    # try:
    #     async_gradio_app = shared.gradio_root
    #     flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
    #     if async_gradio_app.share:
    #         flag += f''' or {async_gradio_app.share_url}'''
    #     print(flag)
    # except Exception as e:
    #     print(e)

    def progressbar(async_task, number, text):
        print(f'[Fooocus] {text}')
        async_task.yields.append(['preview', (number, text, None)])

    def yield_result(async_task, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]

        async_task.results = async_task.results + imgs

        async_task.yields.append(['results', async_task.results])
        return

    def build_image_wall(async_task):
        results = []

        if len(async_task.results) < 2:
            return

        for img in async_task.results:
            if isinstance(img, str) and os.path.exists(img):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return
            results.append(img)

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    def apply_enabled_loras(loras):
        enabled_loras = []
        for lora_model, lora_weight in loras:
            enabled_loras.append([lora_model, lora_weight])

        return enabled_loras

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task: AsyncTask):
        execution_start_time = time.perf_counter()
        async_task.processing = True

        args = async_task.args
        # args.reverse()

        prompt = args.prompt
        negative_prompt = args.negative_prompt
        style_selections = args.style_selections
        performance_selection = Performance(args.performance_selection)
        aspect_ratios_selection = args.aspect_ratios_selection
        image_number = args.image_number
        output_format = args.output_format
        image_seed = args.image_seed
        sharpness = args.sharpness
        guidance_scale = args.guidance_scale
        base_model_name = args.base_model_name
        refiner_model_name = args.refiner_model_name
        refiner_switch = args.refiner_switch
        loras = apply_enabled_loras(args.loras)
        input_image_checkbox = args.uov_input_image is not None or args.inpaint_input_image is not None or len(args.image_prompts) > 0
        current_tab = 'uov' if args.uov_method != flags.disabled else 'ip' if len(args.image_prompts) > 0 else 'inpaint' if args.inpaint_input_image is not None else None
        uov_method = args.uov_method
        uov_input_image = args.uov_input_image
        outpaint_selections = args.outpaint_selections
        inpaint_input_image = args.inpaint_input_image
        inpaint_additional_prompt = args.inpaint_additional_prompt
        inpaint_mask_image_upload = args.inpaint_input_image['mask'] if isinstance(inpaint_input_image, dict) and 'mask' in inpaint_input_image else None

        disable_preview = args.disable_preview
        adm_scaler_positive = args.adm_scaler_positive
        adm_scaler_negative = args.adm_scaler_negative
        adm_scaler_end = args.adm_scaler_end
        adaptive_cfg = args.adaptive_cfg
        sampler_name = args.sampler_name
        scheduler_name = args.scheduler_name
        overwrite_step = args.overwrite_step
        overwrite_switch = args.overwrite_switch
        overwrite_width = args.overwrite_width
        overwrite_height = args.overwrite_height
        overwrite_vary_strength = args.overwrite_vary_strength
        overwrite_upscale_strength = args.overwrite_vary_strength
        mixing_image_prompt_and_vary_upscale = args.mixing_image_prompt_and_vary_upscale
        mixing_image_prompt_and_inpaint = args.mixing_image_prompt_and_inpaint
        debugging_cn_preprocessor = args.debugging_cn_preprocessor
        skipping_cn_preprocessor = args.skipping_cn_preprocessor
        canny_low_threshold = args.canny_low_threshold
        canny_high_threshold = args.canny_high_threshold
        refiner_swap_method = args.refiner_swap_method
        controlnet_softness = args.controlnet_softness
        freeu_enabled = args.freeu_enabled
        freeu_b1 = args.freeu_b1
        freeu_b2 = args.freeu_b2
        freeu_s1 = args.freeu_s1
        freeu_s2 = args.freeu_s2
        debugging_inpaint_preprocessor = args.debugging_inpaint_preprocessor
        inpaint_disable_initial_latent = args.inpaint_disable_initial_latent
        inpaint_engine = args.inpaint_engine
        inpaint_strength = args.inpaint_strength
        inpaint_respective_field = args.inpaint_respective_field
        inpaint_mask_upload_checkbox = inpaint_mask_image_upload is not None
        invert_mask_checkbox = args.invert_mask_checkbox
        inpaint_erode_or_dilate = args.inpaint_erode_or_dilate

        # save_metadata_to_images = args. if not args_manager.args.disable_metadata else False
        # metadata_scheme = MetadataScheme() if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        cn_tasks = {x: [] for x in flags.ip_list}
        for img_prompt in args.image_prompts:
            cn_img, cn_stop, cn_weight, cn_type = img_prompt
            cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        outpaint_selections = [o.lower() for o in outpaint_selections]
        base_model_additional_loras = []
        raw_style_selections = copy.deepcopy(style_selections)
        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        if base_model_name == refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        steps = performance_selection.steps()

        if performance_selection == Performance.EXTREME_SPEED:
            print('Enter LCM mode.')
            progressbar(async_task, 1, 'Downloading LCM components ...')
            loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]

            if refiner_model_name != 'None':
                print(f'Refiner disabled in LCM mode.')

            refiner_model_name = 'None'
            sampler_name = 'lcm'
            scheduler_name = 'lcm'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        print(f'[Parameters] Adaptive CFG = {adaptive_cfg}')
        print(f'[Parameters] Sharpness = {sharpness}')
        print(f'[Parameters] ControlNet Softness = {controlnet_softness}')
        print(f'[Parameters] ADM Scale = '
              f'{adm_scaler_positive} : '
              f'{adm_scaler_negative} : '
              f'{adm_scaler_end}')

        patch_settings[pid] = PatchSettings(
            sharpness,
            adm_scaler_end,
            adm_scaler_positive,
            adm_scaler_negative,
            controlnet_softness,
            adaptive_cfg
        )

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        controlnet_depth_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(image_seed)
        print(f'[Parameters] Seed = {seed}')

        goals = []
        tasks = []
        # input_image_checkbox True
        # current_tab ip
        # mixing_image_prompt_and_inpaint False
        print("input_image_checkbox", input_image_checkbox)
        print("current_tab", current_tab)
        print("mixing_image_prompt_and_inpaint", mixing_image_prompt_and_inpaint)
        if input_image_checkbox:
            if (current_tab == 'uov' or (
                    current_tab == 'ip' and mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        steps = performance_selection.steps_uov()

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
            if (current_tab == 'inpaint' or (
                    current_tab == 'ip' and mixing_image_prompt_and_inpaint)) \
                    and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]

                if inpaint_mask_upload_checkbox:
                    if isinstance(inpaint_mask_image_upload, np.ndarray):
                        if inpaint_mask_image_upload.ndim == 3:
                            H, W, C = inpaint_image.shape
                            inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
                            inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                            inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            inpaint_mask = inpaint_mask_image_upload

                if int(inpaint_erode_or_dilate) != 0:
                    inpaint_mask = erode_or_dilate(inpaint_mask, inpaint_erode_or_dilate)

                if invert_mask_checkbox:
                    inpaint_mask = 255 - inpaint_mask

                inpaint_image = HWC3(inpaint_image)
                print(isinstance(inpaint_image, np.ndarray))
                print(isinstance(inpaint_mask, np.ndarray))
                print(np.any(inpaint_mask > 127))
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.5
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if inpaint_additional_prompt != '':
                        if prompt == '':
                            prompt = inpaint_additional_prompt
                        else:
                            prompt = inpaint_additional_prompt + '\n' + prompt
                    goals.append('inpaint')
            if current_tab == 'ip' or \
                    mixing_image_prompt_and_vary_upscale or \
                    mixing_image_prompt_and_inpaint:
                goals.append('cn')
                progressbar(async_task, 1, 'Downloading control models ...')
                if len(cn_tasks[flags.cn_canny]) > 0:
                    controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(cn_tasks[flags.cn_cpds]) > 0:
                    controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(cn_tasks[flags.cn_depth]) > 0:
                    controlnet_depth_path = modules.config.downloading_controlnet_depth()
                if len(cn_tasks[flags.cn_ip]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
                if len(cn_tasks[flags.cn_ip_face]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')
                progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path, controlnet_depth_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        if overwrite_step > 0:
            steps = overwrite_step

        switch = int(round(steps * refiner_switch))

        if overwrite_switch > 0:
            switch = overwrite_switch

        if overwrite_width > 0:
            width = overwrite_width

        if overwrite_height > 0:
            height = overwrite_height

        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {steps} - {switch}')

        progressbar(async_task, 1, 'Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
                                        loras=loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner)

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []
            
            for i in range(image_number):
                task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not

                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
                task_prompt = apply_wildcards(prompt, task_rng)
                task_prompt = apply_arrays(task_prompt, i)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if use_style:
                    for s in style_selections:
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

        if 'vary' in goals:
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if overwrite_vary_strength > 0:
                denoising_strength = overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = uov_input_image.shape
            progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(uov_input_image)
            print(f'Image upscaled.')

            if '1.5x' in uov_method:
                f = 1.5
            elif '2x' in uov_method:
                f = 2.0
            else:
                f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                print(f'[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in uov_method:
                direct_return = True
            elif image_is_super_large:
                print('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                uov_input_image_path = log(uov_input_image, output_format=output_format)
                yield_result(async_task, uov_input_image_path)
                return

            tiled = True
            denoising_strength = 0.382

            if overwrite_upscale_strength > 0:
                denoising_strength = overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')
        print("goals ", goals)
        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                H, W, C = inpaint_image.shape
                if 'top' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                          constant_values=255)
                if 'bottom' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if 'left' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
                                          constant_values=255)
                if 'right' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
                                          constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                inpaint_strength = 1.0
                inpaint_respective_field = 1.0

            denoising_strength = inpaint_strength
            print("saving inpaint")
            from PIL import Image
            Image.fromarray(inpaint_image).save('inpaint_image.png')
            Image.fromarray(inpaint_mask).save('inpaint_mask.png')
            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )

            if debugging_inpaint_preprocessor:
                yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing())
                return

            progressbar(async_task, 13, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img, canny_low_threshold, canny_high_threshold)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img)
                    return
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img)
                    return
            for task in cn_tasks[flags.cn_depth]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.depth(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img)
                    return
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img)
                    return
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not skipping_cn_preprocessor:
                    cn_img = extras.face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img)
                    return

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                freeu_b1,
                freeu_b2,
                freeu_s1,
                freeu_s2
            )

        all_steps = steps * image_number

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            print('Using lcm scheduler.')

        async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            async_task.yields.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}{ordinal_suffix(current_task_id + 1)} Sampling', y)])

        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

            try:
                if async_task.last_stop is not False:
                    ldm_patched.modules.model_management.interrupt_current_processing()
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, controlnet_canny_path),
                        (flags.cn_cpds, controlnet_cpds_path),
                        (flags.cn_depth, controlnet_depth_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                imgs = pipeline.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
                    steps=steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale,
                    refiner_swap_method=refiner_swap_method,
                    disable_preview=disable_preview
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                imgs_b64 = []
                for x in imgs:
                    # d = [('Prompt', 'prompt', task['log_positive_prompt']),
                    #      ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                    #      ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                    #      ('Styles', 'styles', str(raw_style_selections)),
                    #      ('Performance', 'performance', performance_selection.value)]

                    # if performance_selection.steps() != steps:
                    #     d.append(('Steps', 'steps', steps))

                    # d += [('Resolution', 'resolution', str((width, height))),
                    #       ('Guidance Scale', 'guidance_scale', guidance_scale),
                    #       ('Sharpness', 'sharpness', sharpness),
                    #       ('ADM Guidance', 'adm_guidance', str((
                    #           modules.patch.patch_settings[pid].positive_adm_scale,
                    #           modules.patch.patch_settings[pid].negative_adm_scale,
                    #           modules.patch.patch_settings[pid].adm_scaler_end))),
                    #       ('Base Model', 'base_model', base_model_name),
                    #       ('Refiner Model', 'refiner_model', refiner_model_name),
                    #       ('Refiner Switch', 'refiner_switch', refiner_switch)]

                    # if refiner_model_name != 'None':
                    #     if overwrite_switch > 0:
                    #         d.append(('Overwrite Switch', 'overwrite_switch', overwrite_switch))
                    #     if refiner_swap_method != flags.refiner_swap_method:
                    #         d.append(('Refiner Swap Method', 'refiner_swap_method', refiner_swap_method))
                    # if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
                    #     d.append(('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))

                    # d.append(('Sampler', 'sampler', sampler_name))
                    # d.append(('Scheduler', 'scheduler', scheduler_name))
                    # d.append(('Seed', 'seed', str(task['task_seed'])))

                    # if freeu_enabled:
                    #     d.append(('FreeU', 'freeu', str((freeu_b1, freeu_b2, freeu_s1, freeu_s2))))

                    # for li, (n, w) in enumerate(loras):
                    #     if n != 'None':
                    #         d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))

                    metadata_parser = None
                    # if save_metadata_to_images:
                    #     metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                    #     metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                    #                              task['log_negative_prompt'], task['negative'],
                    #                              steps, base_model_name, refiner_model_name, loras)
                    # d.append(('Metadata Scheme', 'metadata_scheme', metadata_scheme.value if save_metadata_to_images else save_metadata_to_images))
                    # d.append(('Version', 'version', 'Fooocus v' + fooocus_version.version))
                    imgs_b64.append(log(x, output_format))

                yield_result(async_task, imgs_b64)
            except ldm_patched.modules.model_management.InterruptProcessingException as e:
                if async_task.last_stop == 'skip':
                    print('User skipped')
                    async_task.last_stop = False
                    continue
                else:
                    print('User stopped')
                    break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')
        async_task.processing = False
        return

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            print(task)
            try:
                handler(task)
                print("append to finish")
                task.yields.append(['finish', task.results])
                pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.yields.append(['finish', task.results])
            finally:
                if pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[pid]
    pass


threading.Thread(target=worker, daemon=True).start()
print("Inference worker started")