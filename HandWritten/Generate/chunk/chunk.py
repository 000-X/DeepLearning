import numpy as np
from PIL import Image


def split_image(image_path, labels_path, grid_size=(7, 7), save_dir='splits'):
    """
    将图像按照指定的网格尺寸划分，并保存每个切片。

    参数:
    - image_path: 输入图像的路径。
    - grid_size: 划分网格的尺寸（行数，列数）。
    - save_dir: 保存切片的目录。
    """
    img = Image.open(image_path)
    image = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_width, img_height = image.size
    print(f"w, h --> {img_width, img_height}")

    # 确保图像尺寸适合网格划分
    assert img_width % grid_size[0] == 0, "图像宽度必须能被网格宽度整除。"
    assert img_height % grid_size[1] == 0, "图像高度必须能被网格高度整除。"

    tile_width = img_width // grid_size[0]  # 向下取整
    tile_height = img_height // grid_size[1]
    print(f"tile_w_h --> {tile_width, tile_height}")

    # 模拟标签数据，实际使用时需要替换为加载的标签信息
    labels = np.random.randint(0, 10, size=(grid_size[0], grid_size[1]))  # 假设有10个类别

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 计算每个网格的边界
            left = i * tile_width
            upper = j * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # 切片并保存
            crop_img = img.crop((left, upper, right, lower))
            crop_img.save(f"{save_dir}/tile_{i}_{j}.png")

            # 获取并保存标签
            label = labels[j, i]  # 调整索引以匹配标签数组
            print(f"Tile ({i}, {j}) Label: {label}")

    print(labels)


# 示例用法
split_image(r'H:\pro_yzy\HandWritten\Generate\chunk\out\line_10_chunk_0.png', '../../dataset/font1/labels/image_1.json')
