import os

import generate_new

Path = 'font'
output_path = '../dataset'
f = open('txt/Text.txt', 'r', encoding='UTF-8')
txt_path = 'txt/Text.txt'
Texts = f.readlines()

# Font列表
font_paths = []
for dir_path, _, file_names in os.walk(Path):
    for file_name in file_names:
        file_paths = os.path.join(dir_path, file_name)
        font_paths.append(file_paths)

# print(font_paths)
generate_new.generate_dataset(txt_path, font_paths, output_path)
