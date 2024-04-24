import json
import os

from PIL import Image, ImageFont, ImageDraw


def generate_text_image(text_file, font_path, font_size, tile_width, tile_height, image_library, output_path):
    # 加载字体
    font = ImageFont.truetype(font_path, font_size)
    font_name = os.path.basename(font_path).split('.')[0]
    path = os.path.join(output_path, font_name)
    if not os.path.exists(path):
        os.makedirs(path)
    feat = os.path.join(path, 'feature')
    if not os.path.exists(feat):
        os.makedirs(feat)
    labels = os.path.join(path, 'labels')
    if not os.path.exists(labels):
        os.makedirs(labels)
    image_cache = {}

    # 读取文本内容
    with open(text_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines):
        line = line.strip().replace(' ', '').replace('\n', '')
        chunks = [line[i:i + 12] for i in range(0, len(line), 12)]  # 每12个字符分割

        for chunk_index, chunk in enumerate(chunks):
            # 创建画板
            num_tiles = len(chunk)
            canvas = Image.new('RGB', (num_tiles * tile_width, tile_height), 'white')
            draw = ImageDraw.Draw(canvas)
            annotations = {
                'image_name': f'line_{line_number}_chunk_{chunk_index}.png',
                'image_text': chunk,
                'chars': []
            }

            for i, char in enumerate(chunk):
                char_image_path = os.path.join(image_library, f'char_{char}.png')
                if char not in image_cache:
                    if not os.path.exists(char_image_path):
                        # 如果图片库中不存在，则生成
                        char_image = Image.new('RGB', (tile_width, tile_height), 'white')
                        draw_temp = ImageDraw.Draw(char_image)
                        draw_temp.text((0, 0), char, font=font, fill='black')
                        char_image.save(char_image_path)
                        print(f"Generated new image for '{char}'")
                    else:
                        # 加载已存在的图片
                        char_image = Image.open(char_image_path)
                        char_image = char_image.resize((tile_width, tile_height))  # 确保图像符合格子尺寸
                    image_cache[char] = char_image
                else:
                    char_image = image_cache[char]

                # 定位图像到指定的网格位置
                left = i * tile_width
                upper = 0
                canvas.paste(char_image, (left, upper))
                annotations['chars'].append({
                    'char': char,
                    'bbox': [left, upper, left + tile_width, upper + tile_height]
                })

            # 保存生成的文本行图片
            chunk_output_path = os.path.join(feat, f'line_{line_number}_chunk_{chunk_index}.png')
            canvas.save(chunk_output_path)
            # 保存注解信息到 JSON 文件
            annotation_path = os.path.join(labels, f'line_{line_number}_chunk_{chunk_index}_annotations.json')
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=4)


# 使用示例
font_paths = []
for dir_path, _, file_names in os.walk('../font'):
    for file_name in file_names:
        file_paths = os.path.join(dir_path, file_name)
        font_paths.append(file_paths)

for font in font_paths:
    generate_text_image(
        text_file='../txt/Text.txt',
        font_path=font,
        font_size=24,
        tile_width=32,
        tile_height=32,
        image_library='char_images',
        output_path='out'
    )
print("over!")
