import os
import shutil
import random

def collect_and_shuffle_images(category, base_dirs):
    """
    收集指定类别的图片，并打乱顺序
    """
    all_images = []
    original_counts = []  # 用于记录每个文件夹中图片数量

    for base_dir in base_dirs:
        source_dir = os.path.join(base_dir, category)
        images = [img for img in os.listdir(source_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_images.extend([(os.path.join(source_dir, img), img) for img in images])  # 保存全路径和文件名
        original_counts.append(len(images))

    random.shuffle(all_images)  # 打乱所有图片
    return all_images, original_counts

def distribute_images(images, counts, target_dirs, category):
    """
    按照指定数量将图片分配到目标文件夹中
    """
    start_idx = 0
    for target_dir, count in zip(target_dirs, counts):
        target_category_dir = os.path.join(target_dir, category)
        os.makedirs(target_category_dir, exist_ok=True)

        for img_path, img_name in images[start_idx:start_idx + count]:
            shutil.copy(img_path, os.path.join(target_category_dir, img_name))
        
        start_idx += count

def process_categories(categories, source_dirs, target_dirs):
    """
    对所有类别的数据进行收集、打乱和重新分配
    """
    for category in categories:
        print(f"Processing category: {category}")
        
        # 收集和打乱数据
        all_images, original_counts = collect_and_shuffle_images(category, source_dirs)
        
        # 按比例重新分配数据
        distribute_images(all_images, original_counts, target_dirs, category)

def main():
    random.seed(42)  # 设置随机种子以保证可重复性

    # 原始数据集目录
    source_dirs = ['./rps', './rps-test-set']
    # 新的数据集目录
    target_dirs = ['./rps_new', './rps-test-set_new']
    # 类别
    categories = ['paper', 'rock', 'scissors']

    # 处理数据集
    process_categories(categories, source_dirs, target_dirs)

    print("Dataset shuffling and reallocation completed!")

if __name__ == "__main__":
    main()
