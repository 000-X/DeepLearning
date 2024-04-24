import json
import os

from PIL import Image, ImageDraw, ImageFont


def generate_dataset(texts, font_paths, output_dir, margin=30, max_chars_per_line=12):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    font_path_txt = os.path.join(output_dir, 'font_path.txt')
    paths = []

    for font_path in font_paths:
        font_size = 24
        font_name = os.path.basename(font_path).split('.')[0]
        font_output_dir = os.path.join(output_dir, font_name)
        absolute_path = os.path.abspath(font_output_dir)
        paths.append(absolute_path)
        feature_dir = os.path.join(font_output_dir, 'feature')
        labels_dir = os.path.join(font_output_dir, 'labels')

        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        flag = 0
        for i, line in enumerate(texts):
            line = line.replace(' ', '').replace('\n', '')
            for start in range(0, len(line), max_chars_per_line):
                filtered_line = line[start:start + max_chars_per_line]
                if not filtered_line:
                    continue

                font = ImageFont.truetype(font_path, font_size)
                image = Image.new('RGB', (1, 1), 'white')
                draw = ImageDraw.Draw(image)

                overall_bbox = draw.textbbox((0, 0), filtered_line, font=font)
                text_width, text_height = overall_bbox[2] - overall_bbox[0], overall_bbox[3] - overall_bbox[1]

                image_width = text_width + 2 * margin
                image_height = text_height + 2 * margin

                image = Image.new('RGB', (image_width, image_height), 'white')
                draw = ImageDraw.Draw(image)
                draw.text((margin, margin), filtered_line, fill='black', font=font)

                file_base_name = f"image_{flag + 1}"
                image_path = os.path.join(feature_dir, f"{file_base_name}.png")
                json_path = os.path.join(labels_dir, f"{file_base_name}.json")
                image.save(image_path)

                annotations = {'image_name': f"{file_base_name}.png", 'image_text': filtered_line, 'chars': []}

                current_w = margin
                for j, char in enumerate(filtered_line):
                    char_bbox = draw.textbbox((current_w, margin), char, font=font)
                    annotations['chars'].append({
                        'char_index': j,
                        'text_char': char,
                        'bbox': [char_bbox[0], char_bbox[1], char_bbox[2], char_bbox[3]]
                    })
                    current_w += char_bbox[2] - char_bbox[0]

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=4)

                flag += 1

    with open(font_path_txt, 'w', encoding='UTF-8') as f:
        for path in paths:
            f.write(path + '\n')

    print("Dataset generation complete.")
