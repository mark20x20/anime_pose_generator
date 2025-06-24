import gradio as gr
import os
import glob
import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
import deepdanbooru as dd
import tensorflow as tf
import numpy as np

# ===== 初期設定 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
ci_config = Config(clip_model_name="ViT-L-14/openai", device=device)
interrogator = Interrogator(ci_config)

danbooru_model_path = "C:/AI/anime_pose_generator/models/deepdanbooru_model/pretrained_model"
dd_model = dd.project.load_model_from_project(danbooru_model_path, compile_model=True)
with open(os.path.join(danbooru_model_path, 'tags.txt'), 'r', encoding='utf-8') as f:
    danbooru_tags = [line.strip() for line in f.readlines()]

unwanted_keywords = [
    "from wikipedia", "school curriculum expert", "flume cover art", "spritesheet",
    "crimes", "femme", "victor", "we", "quinn", "alpaca", "low res", "behold", "wife",
    "belle delphine", "elon musk", "subaru", "vargas", "magali villeneuve", "esa", 
    "derg", "by An Zhengwen", "yee chong silverfox", "ozabu", "Miyagawa Shunsui",
    "3d", "photo_(medium)", "photo_background", "photo_inset", "blur_censor", "blurry", 
    "blurry_background", "blurry_foreground", "chromatic_aberration", "film_grain",
    "depth_of_field", "motion_blur", "focused", "reference_inset", "jail", "prisoner", 
    "queen in a glass prison", "with prison clothing", "leaked image", 
    "official splash art", "image on the store website", "google images", "twitter pfp",
    "true realistic image", "perfect dynamic body form", "the face of absurdly beautiful", 
    "magic spell icon", "arknights", "dungeondraft", "d&d monster", "nagas", 
    "ophelia", "dionysus", "cleric", "slimy tongue", "eats bambus"
    ]

    

# ===== ヘルパー関数 =====
def clean_prompt(prompt_line):
    tags = [tag.strip() for tag in prompt_line.split(",")]
    unique_tags = set()
    cleaned_tags = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in unwanted_keywords or tag in unique_tags or tag == '':
            continue
        unique_tags.add(tag)
        cleaned_tags.append(tag)
    return ", ".join(cleaned_tags)

def get_danbooru_tags(image_path, top_k, threshold):
    image_array = dd.data.load_image_for_evaluate(image_path, width=512, height=512)
    predictions = dd_model.predict(np.expand_dims(image_array, axis=0))[0]
    tag_scores = [(tag, score) for tag, score in zip(danbooru_tags, predictions) if score >= threshold]
    tag_scores.sort(key=lambda x: x[1], reverse=True)
    top_tags = [tag for tag, _ in tag_scores[:top_k]]
    return ', '.join(top_tags)

def get_clip_tags(image, top_k):
    ci_caption = interrogator.interrogate(image)
    tags = [tag.strip() for tag in ci_caption.split(",")]
    return ', '.join(tags[:top_k])

def generate_captions(image_folder, clip_limit, danbooru_limit, threshold, negative_prompt):
    if not os.path.exists(image_folder):
        return "入力フォルダが存在しません。"

    character_name = os.path.basename(image_folder).strip()
    trigger_word = f"<{character_name}>"
    image_paths = glob.glob(os.path.join(image_folder, "*.png"))

    result_logs = ""
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        clip_caption = get_clip_tags(image, top_k=clip_limit)
        danbooru_caption = get_danbooru_tags(img_path, top_k=danbooru_limit, threshold=threshold)
        combined_caption = f"{trigger_word}, {clip_caption}, {danbooru_caption}"
        final_caption = clean_prompt(combined_caption)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        with open(os.path.join(image_folder, f"{base_name}.txt"), "w", encoding="utf-8") as f:
            f.write(final_caption + "\n")
            if negative_prompt.strip() != "":
                f.write(f"### Negative Prompt: {negative_prompt.strip()}\n")

        result_logs += f"[{base_name}]\n{final_caption}\n"
        if negative_prompt.strip() != "":
            result_logs += f"NEG: {negative_prompt.strip()}\n\n"
        else:
            result_logs += "\n"

    return f" {len(image_paths)}件のキャプション生成が完了しました。\n\n{result_logs}"


# ===== キャプション個別編集用 関数 =====
def load_image_list(folder_path):
    folder_path = folder_path.strip(' "')
    if not os.path.exists(folder_path):
        return gr.update(choices=[])
    images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    return gr.update(choices=images)

def load_caption_text(folder_path, image_name):
    if not image_name:
        return ""
    txt_path = os.path.join(folder_path, os.path.splitext(image_name)[0] + ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def save_caption_text(folder_path, image_name, new_caption):
    if not image_name:
        return "画像を選択してください。"
    txt_path = os.path.join(folder_path, os.path.splitext(image_name)[0] + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(new_caption)
    return f" {image_name} のプロンプトを保存しました。"

def apply_negative_prompt_ui(folder_path, negative_prompt):
    folder_path = folder_path.strip(' "')
    if not os.path.exists(folder_path):
        return "入力フォルダが存在しません。"

    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        return "テキストファイルが見つかりません。"

    count = 0
    for txt_file in txt_files:
        with open(txt_file, "r+", encoding="utf-8") as f:
            content = f.read()
            # すでにネガティブプロンプトがある場合はスキップ
            if "### Negative Prompt:" in content:
                continue  
            f.write(f"\n### Negative Prompt: {negative_prompt.strip()}\n")
            count += 1

    return f"ネガティブプロンプトを {count} 件のファイルに追加しました。"




##python scripts/utils/caption_generator.py
