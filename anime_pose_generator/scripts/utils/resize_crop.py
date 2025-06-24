import os
from PIL import Image, ImageOps

def resize_and_pad_images(folder_path, output_path, target_size=512, bg_color=(255, 255, 255)):
    if not os.path.isdir(folder_path):
        return "入力フォルダが存在しません。"
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        return "画像ファイルが見つかりませんでした。"

    prefix = os.path.basename(os.path.normpath(folder_path))

    for idx, image_file in enumerate(images, start=1):
        img_path = os.path.join(folder_path, image_file)
        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size

            # 正方形サイズを決定（大きい方に合わせる）
            square_size = max(width, height)

            # 正方形のキャンバスを作成（背景色で塗りつぶす）
            new_img = Image.new("RGB", (square_size, square_size), bg_color)

            # 画像を中央に配置
            left = (square_size - width) // 2
            top = (square_size - height) // 2
            new_img.paste(img, (left, top))

            # 最終リサイズ
            final_img = new_img.resize((target_size, target_size), Image.LANCZOS)

            filename = f"{prefix}_{idx:03}.png"
            save_path = os.path.join(output_path, filename)
            final_img.save(save_path)

        except Exception as e:
            return f"エラー発生: {e}"

    return f"リサイズ完了！{len(images)} 枚の画像を {output_path} に保存しました。"


# ===== リサイズ処理 =====
def resize_images_ui(folder_path, output_path, slider_size, custom_size_input, use_custom):
    folder_path = folder_path.strip(' "')
    output_path = output_path.strip(' "')

    if not folder_path or not os.path.exists(folder_path):
        return "入力フォルダパスを正しく入力してください。"
    if not output_path:
        return "出力フォルダパスを正しく入力してください。"

    try:
        size = int(custom_size_input) if use_custom else int(slider_size)
    except ValueError:
        return "サイズ指定が無効です。数値を確認してください。"

    result = resize_and_pad_images(folder_path, output_path, target_size=size)
    return result

##python scripts\utils\resize_crop.py
