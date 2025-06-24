import gradio as gr
import os
import subprocess
import glob
import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
import deepdanbooru as dd
import tensorflow as tf
import numpy as np

from scripts.utils.resize_crop import resize_images_ui, resize_and_pad_images
from scripts.utils.image_generation import (
    generate_images,
    generate_images_with_controlnet_lora,
    save_selected_images,
    process_base_model_input,
    process_lora_inputs,
    debug_lora_inputs
)
from scripts.utils.caption_generator import (
    generate_captions as generate_captions_ui,
    load_image_list,
    load_caption_text,
    save_caption_text,
    apply_negative_prompt_ui
)


# ===== 共通モデル・設定初期化 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
ci_config = Config(clip_model_name="ViT-L-14/openai", device=device)
interrogator = Interrogator(ci_config)

danbooru_model_path = "C:/AI/anime_pose_generator/models/deepdanbooru_model/pretrained_model"
dd_model = dd.project.load_model_from_project(danbooru_model_path, compile_model=True)
with open(os.path.join(danbooru_model_path, 'tags.txt'), 'r', encoding='utf-8') as f:
    danbooru_tags = [line.strip() for line in f.readlines()]

unwanted_keywords = ["from wikipedia", "spritesheet", "low res", "blurry", "blur_censor", "twitter pfp", "photo_inset", "motion_blur", "chromatic_aberration", "focused"]


# ===== マップ作成処理 =====
def generate_maps_ui(folder_path, output_path, cam_x, cam_y, cam_z, rot_x, rot_y, rot_z, start_frame, end_frame, step, skeleton, depth, normal):
    folder_path = folder_path.strip(' "')
    output_path = output_path.strip(' "')

    if not os.path.exists(folder_path):
        return "入力フォルダが存在しません。"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    maps = []
    if skeleton:
        maps.append("Skeleton")
    if depth:
        maps.append("Depth")
    if normal:
        maps.append("Normal")

    if not maps:
        return "少なくとも1つのマップタイプを選択してください。"

    blender_cmd = [
        "blender", "-b", "--python", "scripts/blender_generate_maps.py", "--",
        f"--input={folder_path}",
        f"--output={output_path}",
        f"--cam_pos={cam_x},{cam_y},{cam_z}",
        f"--cam_rot={rot_x},{rot_y},{rot_z}",
        f"--frames={start_frame},{end_frame},{step}",
        f"--maps={','.join(maps)}"
    ]

    try:
        subprocess.run(blender_cmd, check=True)
        return "マップ生成処理が完了しました。"
    except subprocess.CalledProcessError as e:
        return f"エラー発生: {e}"


# ===== Gradio UI =====
with gr.Blocks() as app:
    gr.Markdown("## 統合AIツール")

    # === 画像リサイズ ===
    with gr.Tab("画像リサイズ"):
        folder_path_input = gr.Textbox(label="入力フォルダパス")
        output_path_input = gr.Textbox(label="出力フォルダパス")
        image_size_slider = gr.Slider(256, 1024, value=512, step=64, label="標準サイズ")
        with gr.Row():
            use_custom_checkbox = gr.Checkbox(label="カスタムサイズを使用")
            custom_size_input = gr.Number(label="カスタムサイズ (px)", value=512, precision=0)
        resize_result = gr.Textbox(label="処理結果", lines=4)
        resize_execute_btn = gr.Button("リサイズ実行")
        resize_execute_btn.click(
            fn=resize_images_ui,
            inputs=[folder_path_input, output_path_input, image_size_slider, custom_size_input, use_custom_checkbox],
            outputs=resize_result
        )

    # === マップ作成 ===
    with gr.Tab("マップ作成"):
        map_folder_input = gr.Textbox(label="入力フォルダパス (FBXファイル)")
        map_output_input = gr.Textbox(label="出力フォルダパス")
        with gr.Row():
            cam_x = gr.Number(label="X 座標", value=0.0)
            cam_y = gr.Number(label="Y 座標", value=-5.0)
            cam_z = gr.Number(label="Z 座標", value=2.0)
        with gr.Row():
            rot_x = gr.Number(label="回転 X (Pitch)", value=74.4)
            rot_y = gr.Number(label="回転 Y (Yaw)", value=0.0)
            rot_z = gr.Number(label="回転 Z (Roll)", value=-12.8)
        with gr.Row():
            start_frame = gr.Number(label="開始フレーム", value=1, precision=0)
            end_frame = gr.Number(label="終了フレーム", value=30, precision=0)
            step = gr.Number(label="フレーム間隔", value=1, precision=0)
        with gr.Row():
            skeleton_map = gr.Checkbox(label="Skeleton")
            depth_map = gr.Checkbox(label="Depth")
            normal_map = gr.Checkbox(label="Normal")
        map_result = gr.Textbox(label="処理結果", lines=4)
        map_execute_btn = gr.Button("マップ作成実行")
        map_execute_btn.click(
            fn=generate_maps_ui,
            inputs=[map_folder_input, map_output_input, cam_x, cam_y, cam_z, rot_x, rot_y, rot_z, start_frame, end_frame, step, skeleton_map, depth_map, normal_map],
            outputs=map_result
        )

        # === キャプション生成タブ内に統合 ===
    with gr.Tab("キャプション生成"):
        caption_folder_input = gr.Textbox(label="画像フォルダパス")
        with gr.Row():
            clip_limit = gr.Number(label="CLIP タグ数", value=3, precision=0)
            danbooru_limit = gr.Number(label="Danbooru タグ数", value=7, precision=0)
            danbooru_threshold = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Danbooru スコア閾値")
        caption_result = gr.Textbox(label="処理結果とキャプション例", lines=15)
        caption_execute_btn = gr.Button("キャプション生成実行")

        # キャプション生成 実行ボタン
        caption_execute_btn.click(
            fn=generate_captions_ui,
            inputs=[caption_folder_input, clip_limit, danbooru_limit, danbooru_threshold],
            outputs=[caption_result]
        )

        # ===== キャプション個別編集 (この位置に統合) =====
        gr.Markdown("### キャプション個別編集")
        with gr.Row():
            selected_image = gr.Dropdown(label="編集対象画像", choices=[], interactive=True)
            load_images_btn = gr.Button(" 画像一覧読み込み")
        
        caption_editor = gr.Textbox(label="プロンプト編集", lines=5)
        save_caption_btn = gr.Button("プロンプト保存")

        load_images_btn.click(
            fn=load_image_list,
            inputs=[caption_folder_input],
            outputs=[selected_image]
        )
        selected_image.change(
            fn=load_caption_text,
            inputs=[caption_folder_input, selected_image],
            outputs=[caption_editor]
        )
        save_caption_btn.click(
            fn=save_caption_text,
            inputs=[caption_folder_input, selected_image, caption_editor],
            outputs=[caption_result]
        )

                # ===== ネガティブプロンプト一括設定 =====
        gr.Markdown("### ネガティブプロンプト一括設定")
        negative_prompt_input = gr.Textbox(
            label="ネガティブプロンプト（カンマ区切り）", 
            placeholder="例: bad hands, blurry, lowres", 
            lines=2
        )

        caption_execute_btn.click(
            fn=generate_captions_ui,
            inputs=[caption_folder_input, clip_limit, danbooru_limit, danbooru_threshold, negative_prompt_input],
            outputs=[caption_result]
        )

        negative_prompt_apply_btn = gr.Button("ネガティブプロンプト一括適用")

        negative_prompt_apply_btn.click(
            fn=apply_negative_prompt_ui,
            inputs=[caption_folder_input, negative_prompt_input],
            outputs=[caption_result]
        )




    # === LoRA学習タブ ===
    with gr.Tab("LoRA学習"):
        gr.Markdown("### LoRA 学習コントロール")

        with gr.Row():
            lora_data_input = gr.Textbox(label="学習データフォルダパス")
            lora_output_input = gr.Textbox(label="出力フォルダパス")

        lora_base_model_input = gr.Textbox(label="ベースモデルパス")

        with gr.Row():
            lora_precision_input = gr.Dropdown(
                choices=["bf16", "fp16", "fp32"], 
                value="bf16", 
                label="Mixed Precision"
            )
            lora_steps_input = gr.Number(label="最大トレーニングステップ数", value=1000, precision=0)

        with gr.Row():
            lora_rank_input = gr.Number(label="Network Rank", value=64, precision=0)
            lora_alpha_input = gr.Number(label="Network Alpha", value=32, precision=0)

        with gr.Row():
            lora_lr_input = gr.Textbox(label="Learning Rate", value="1e-4")
            lora_clip_skip_input = gr.Dropdown(
                choices=["1", "2"], 
                value="2", 
                label="Clip Skip"
            )

        with gr.Row():
            lora_output_name_input = gr.Textbox(label="出力ファイル名 (Output Name)", value="lora_style")

        lora_result = gr.Textbox(label="学習ログ / 結果", lines=10)
        lora_train_btn = gr.Button("学習開始")

        def start_lora_training(data_folder, output_folder, base_model, precision, steps, rank, alpha, lr, clip_skip, output_name, keep_n):
            data_folder = data_folder.strip(' "')
            output_folder = output_folder.strip(' "')
            base_model = base_model.strip(' "')

            if not os.path.exists(data_folder):
                return "データフォルダが存在しません。"
            if not os.path.exists(base_model):
                return "ベースモデルが存在しません。"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            train_script_path = "C:\\AI\\anime_pose_generator\\kohya_ss\\sd-scripts\\train_network.py"

            train_cmd = [
                "accelerate", "launch", train_script_path,
                f"--pretrained_model_name_or_path={base_model}",
                f"--train_data_dir={data_folder}",
                f"--output_dir={output_folder}",
                f"--max_train_steps={int(steps)}",
                f"--network_dim={int(rank)}",
                f"--network_alpha={int(alpha)}",
                f"--learning_rate={lr}",
                f"--clip_skip={clip_skip}",
                f"--output_name={output_name}",
                f"--mixed_precision={precision}",
                f"--resolution=512,512",
                "--save_precision=bf16",
                "--network_module=networks.lora"
            ]

            try:
                subprocess.run(train_cmd, check=True)
                return "LoRA学習が正常に完了しました。"
            except subprocess.CalledProcessError as e:
                return f"学習中にエラーが発生しました: {e}"

        lora_train_btn.click(
            fn=start_lora_training,
            inputs=[
                lora_data_input, lora_output_input, lora_base_model_input, 
                lora_precision_input, lora_steps_input, lora_rank_input, lora_alpha_input, 
                lora_lr_input, lora_clip_skip_input, lora_output_name_input
            ],
            outputs=[lora_result]
        )
    
    #=画像生成=
    with gr.Tab("画像生成"):
        gr.Markdown("###画像生成ツール")

        # モデル設定
        with gr.Row():
            base_model_path = gr.File(label="ベースモデルファイル (.safetensors)", file_types=[".safetensors"])
            lora_model_paths = gr.Textbox(
                label="LoRAモデルファイルパス（複数はカンマ区切り）",
                placeholder="例: C:/path/to/lora1.safetensors, C:/path/to/lora2.safetensors"
            )
            lora_scale = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="LoRA スケール")

        # プロンプト設定
        positive_prompt = gr.Textbox(label="ポジティブプロンプト", lines=2)
        negative_prompt = gr.Textbox(label="ネガティブプロンプト", lines=2)

        # 生成パラメータ
        with gr.Row():
            steps = gr.Number(label="ステップ数", value=30, precision=0)
            width = gr.Number(label="幅 (px)", value=512, precision=0)
            height = gr.Number(label="高さ (px)", value=512, precision=0)
            seed = gr.Number(label="シード値 (0でランダム)", value=0, precision=0)

        # マップ使用有無
        use_map = gr.Checkbox(label="ControlNetマップを使う", value=False)

        # マップ設定（使用時のみ有効）
        with gr.Row():
            controlnet_model_path = gr.Textbox(label="ControlNetモデルパス", placeholder="例: C:/controlnet-depth")
            map_type = gr.Radio(choices=["depth", "normal", "skeleton"], label="使用マップタイプ", value="depth")
            map_image_path = gr.Textbox(label="マップ画像ファイルパス", placeholder="例: C:/maps/depth0001.png")

        # 保存フォルダ設定
        output_dir = gr.Textbox(label="保存先フォルダパス")

        # 操作ボタン
        with gr.Row():
            generate_button = gr.Button("画像生成")
            save_button = gr.Button("選択画像を保存")

        # 出力ギャラリー
        generated_images = gr.Gallery(label="生成結果", columns=2)

        # 画像生成関数の分岐（ControlNet使用 or 通常生成）
        def generate_router(
            base_model, lora_paths_str, lora_scale,
            prompt, negative_prompt, steps, width, height, seed,
            use_map, controlnet_model_path, map_type, map_image_path
        ):
            base_model_path = process_base_model_input(base_model)
            lora_list = process_lora_inputs(lora_paths_str)

            if use_map:
                return generate_images_with_controlnet_lora(
                    base_model_path, controlnet_model_path, map_image_path, map_type,
                    prompt, negative_prompt, steps, width, height, seed
                )
            else:
                return generate_images(
                    base_model_path, lora_list, lora_scale,
                    prompt, negative_prompt, steps, width, height, seed
                )

        # 生成ボタンに紐付け
        generate_button.click(
            fn=generate_router,
            inputs=[
                base_model_path, lora_model_paths, lora_scale,
                positive_prompt, negative_prompt, steps, width, height, seed,
                use_map, controlnet_model_path, map_type, map_image_path
            ],
            outputs=[generated_images]
        )

        save_button.click(
            fn=save_selected_images,
            inputs=[generated_images, output_dir],
            outputs=[]
        )

        test_button = gr.Button("LoRA受け取りテスト")
        debug_output = gr.Textbox(label="LoRAファイルリスト")

        test_button.click(
            fn=debug_lora_inputs,
            inputs=[lora_model_paths],
            outputs=[debug_output]
        )




app.launch()


## python web_ui_gradio.py