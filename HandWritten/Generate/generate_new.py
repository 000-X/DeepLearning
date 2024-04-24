import json
import os

from PIL import Image, ImageDraw, ImageFont


def generate_dataset(text_file, font_paths, output_dir, margin=20, max_chars_per_line=18):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    font_path_txt = os.path.join(output_dir, 'font_path.txt')
    paths = []

    with open(text_file, 'r', encoding='utf-8') as file:
        paragraphs = file.read().split('\n\n')  # Split text into paragraphs separated by double newlines

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

        font = ImageFont.truetype(font_path, font_size)
        for index, paragraph in enumerate(paragraphs):
            lines = paragraph.split('\n')
            processed_lines = []

            for line in lines:
                start = 0
                while start < len(line):
                    end = start + max_chars_per_line
                    if end < len(line) and not line[end].isspace():
                        end = line.rfind(' ', start, end) + 1 or end
                    processed_lines.append(line[start:end].strip())
                    start = end

            # Calculate the image dimensions based on the processed lines
            max_line_width = max(
                font.getbbox(line)[2] for line in processed_lines) + 2 * margin if processed_lines else 0
            image_height = sum(font.getbbox(line)[3] for line in processed_lines) + (len(processed_lines) + 1) * margin

            image = Image.new('RGB', (max_line_width, image_height), 'white')
            draw = ImageDraw.Draw(image)
            current_h = margin
            text = paragraph.split(' ')
            annotations = {'image_name': f"paragraph_{index + 1}.png", 'image_text': paragraph, 'chars': []}

            for line in processed_lines:
                current_w = margin
                draw.text((current_w, current_h), line, fill='black', font=font)
                for j, char in enumerate(line):
                    char_bbox = draw.textbbox((current_w, current_h), char, font=font)
                    if char not in ' \n':
                        annotations['chars'].append({
                            'char_index': j,
                            'text_char': char,
                            'bbox': [char_bbox[0], char_bbox[1], char_bbox[2], char_bbox[3]]
                        })
                    current_w += char_bbox[2] - char_bbox[0]
                current_h += font.getbbox(line)[3] + margin

            image_path = os.path.join(feature_dir, f"paragraph_{index + 1}.png")
            json_path = os.path.join(labels_dir, f"paragraph_{index + 1}.json")
            image.save(image_path)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=4)

    with open(font_path_txt, 'w', encoding='UTF-8') as f:
        for path in paths:
            f.write(path + '\n')

    print("Dataset generation complete.")

# Example usage
# generate_dataset(
#     text_file='text.txt',
#     font_paths=['path/to/font.ttf'],
#     output_dir='output_directory'
# )
