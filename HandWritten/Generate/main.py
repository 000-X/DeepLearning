import os

from generate_new import generate_dataset


def find_font_files(directory):
    font_paths = []
    supported_extensions = ('.ttf', '.otf')  # 支持的字体文件扩展名
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(supported_extensions):
                file_path = os.path.join(dirpath, filename)
                font_paths.append(file_path)
    return font_paths


def find_txt_files(dir):
    txt_files_paths = []
    supported_extensions = '.txt'
    for path, _, filenames in os.walk(dir):
        for name in filenames:
            if name.endswith(supported_extensions):
                files_path = os.path.join(path, name)
                txt_files_paths.append(files_path)
    return txt_files_paths


def main():
    path = 'font'
    output_path = '../dataset'

    # 查找字体文件
    font_paths = find_font_files(path)
    txt_paths = find_txt_files('txt/image_to_txt')

    # # 如果需要，打印字体路径进行检查
    # for font_path in font_paths:
    #     print(font_path)

    # 调用函数生成数据集
    generate_dataset('txt/image_to_txt/Text.txt', font_paths, output_path)


if __name__ == "__main__":
    main()
