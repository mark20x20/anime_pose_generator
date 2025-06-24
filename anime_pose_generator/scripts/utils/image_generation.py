import os
import torch
import random
import hashlib
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)

# ========= パス処理ユーティリティ =========
def process_base_model_input(file_obj):
    return file_obj.name if hasattr(file_obj, "name") else file_obj

def process_lora_inputs(lora_input_str):
    if not lora_input_str:
        return []
    return [path.strip() for path in lora_input_str.split(",") if path.strip()]

# ========= 条件画像読み込み =========
def load_condition_image(image_path: str, mode: str) -> Image.Image:
    raw = cv2.imread(image_path)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    if mode == "depth":
        processed = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    elif mode == "normal":
        processed = raw  # RGBそのまま
    elif mode == "skeleton":
        processed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    else:
        raise ValueError("Unsupported map mode")

    pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    return pil.resize((512, 512))

# ========= 通常生成 =========
def load_pipeline(base_model_path, lora_model_paths=None, lora_scale=1.0):
    base_model_path = base_model_path.strip('"')
    if not os.path.exists(base_model_path):
        raise ValueError(f"ベースモデルファイルが見つかりません: {base_model_path}")

    pipe = StableDiffusionPipeline.from_single_file(
        base_model_path, torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if lora_model_paths:
        for lora_path in lora_model_paths:
            lora_file = lora_path.name if hasattr(lora_path, "name") else lora_path
            if os.path.exists(lora_file):
                try:
                    pipe.load_lora_weights(lora_file)
                    pipe.fuse_lora(lora_scale)
                except Exception as e:
                    print(f"LoRA適用中にエラー: {e}")
    return pipe

def generate_images(base_model_path, lora_model_path, lora_scale,
                    positive_prompt, negative_prompt, steps, width, height, seed):
    if seed == 0:
        seed = random.randint(1, 999999)
    generator = torch.manual_seed(seed)

    pipe = load_pipeline(base_model_path, lora_model_path, lora_scale)
    output = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        width=int(width),
        height=int(height),
        generator=generator
    )
    return output.images

# ========= ControlNet生成（LoRAあり） =========
def generate_images_with_controlnet_lora(
    base_model_path: str,
    controlnet_model_path: str,
    map_image_path: str,
    map_type: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    lora_model_paths=None,
    lora_scale=1.0
):
    base_model_path = base_model_path.strip('"')
    controlnet_model_path = controlnet_model_path.strip('"')

    if seed == 0:
        seed = random.randint(1, 999999)
    generator = torch.manual_seed(seed)

    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_single_file(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # LoRA適用（if any）
    if lora_model_paths:
        for lora_path in lora_model_paths:
            lora_file = lora_path.name if hasattr(lora_path, "name") else lora_path
            if os.path.exists(lora_file):
                try:
                    pipe.load_lora_weights(lora_file)
                    pipe.fuse_lora(lora_scale)
                except Exception as e:
                    print(f"LoRA適用中にエラー: {e}")

    cond_img = load_condition_image(map_image_path, map_type)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=cond_img,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator
    )
    return output.images

# ========= 保存処理 =========
def save_selected_images(selected_images, output_dir):
    import base64
    from io import BytesIO

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_files = []
    for img_data in selected_images:
        if isinstance(img_data, tuple):
            img_data = img_data[0]

        if isinstance(img_data, Image.Image):
            img = img_data
        elif isinstance(img_data, str) and img_data.startswith("data:image"):
            header, encoded = img_data.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(encoded)))
        else:
            img = Image.open(img_data)

        img_hash = hashlib.sha1(img.tobytes()).hexdigest()[:8]
        filename = f"generated_{img_hash}.png"
        save_path = os.path.join(output_dir, filename)
        img.save(save_path)
        saved_files.append(save_path)

    return f"✅ {len(saved_files)}枚の画像を保存しました：\n" + "\n".join(saved_files)

def debug_lora_inputs(lora_files):
    if not lora_files:
        return "受け取ったファイルはありません。"
    if not isinstance(lora_files, list):
        lora_files = [lora_files]

    files = [getattr(f, "name", f) for f in lora_files]
    return f"受け取ったファイル: {files}"