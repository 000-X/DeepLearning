def extract_and_save_characters(input_file, output_file, num_chars=4000):
    # 打开输入文件读取内容
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # 查找 "var hanzipinlv =" 后的内容
    prefix = "var hanzipinlv ="
    start_index = content.find(prefix)
    if start_index == -1:
        print("未找到指定的前缀。")
        return
    start_index += len(prefix)

    # 提取后续的字符串内容，并只保留前4000个字符
    character_string = content[start_index:start_index + num_chars].strip(' ";\n')

    # 将每个字符保存到输出文件，每行一个字符
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for char in character_string[:num_chars]:
            out_file.write(f"{char}\n")

    print(f"已成功提取并保存前{num_chars}个字符到{output_file}。")


def Sample_save_Characters(sample_path, output_file_path):
    with open(sample_path, 'r', encoding='UTF-8') as f:
        chars = f.read()
    with open(output_file_path, 'a', encoding='UTF-8') as f:
        for char in chars:
            if char.strip():  # 检查字符是否不是空白字符
                f.write(char + '\n')  # 将字符写入目标文件，每个字符后跟一个换行符


# 输入文件路径和输出文件路径
input_file_path = 'txt/zifuji.txt'
output_file_path = 'txt/Characters.txt'
sample_path = 'txt/Sample.txt'
# 调用函数
extract_and_save_characters(input_file_path, output_file_path, 2000)
Sample_save_Characters(sample_path, output_file_path)
