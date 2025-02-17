import os
import random
from PIL import Image, ImageOps
import numpy as np

# 定义数据增强函数
def augment_image(image):
    """
    对图片进行数据增强，随机应用一种或两种增强方式。
    
    参数：
    - image: PIL.Image 对象
    
    返回：
    - augmented_image: 增强后的图片
    """
    enhanced_images = []  # 用于存储增强后的图片
    # 如果图片是 RGBA 模式，转换为 RGB
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # 随机选择增强方式
    augmentations = []
    if random.random() < 0.5:
        # 随机旋转
        angle = random.randint(-15, 15)  # 控制旋转角度在 [-15, 15] 范围内
        augmentations.append(lambda img: img.rotate(angle))
    
    if random.random() < 0.5:
        # 随机水平或垂直翻转
        augmentations.append(lambda img: ImageOps.mirror(img) if random.random() < 0.5 else ImageOps.flip(img))
    
    if random.random() < 0.5:
        # 随机裁切
        def random_crop(img):
            width, height = img.size
            left = random.randint(0, int(0.05 * width))  # 控制裁切范围小一些
            top = random.randint(0, int(0.05 * height))
            right = random.randint(int(0.95 * width), width)
            bottom = random.randint(int(0.95 * height), height)
            return img.crop((left, top, right, bottom)).resize((width, height))
        augmentations.append(random_crop)
    
    if random.random() < 0.5:
        # 随机调整对比度
        augmentations.append(lambda img: ImageOps.autocontrast(img))

    # 随机应用增强方式（最多应用两种增强方式）
    random.shuffle(augmentations)
    for augment in augmentations[:2]:  # 每次随机选两种
        image = augment(image)
        # 将增强后的图片加入列表
        enhanced_images.append(image)
    
    return enhanced_images


def process_category(category, source_dir, target_dir, augmentations=5):
    """
    对指定类别的图片进行数据增强并保存。
    
    参数：
    - category: str，类别名称（如 'paper'）。
    - source_dir: str，源目录路径。
    - target_dir: str，目标目录路径。
    - augmentations: int，每张图片生成的增强图片数量。
    """
    source_category_dir = os.path.join(source_dir, category)
    target_category_dir = os.path.join(target_dir, category)
    os.makedirs(target_category_dir, exist_ok=True)

    # 遍历源目录中的图片
    for img_name in os.listdir(source_category_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 打开图片
            img_path = os.path.join(source_category_dir, img_name)
            image = Image.open(img_path)

            # 对图片进行数据增强
            augmented_images = augment_image(image)

            # 保存原图和增强图片
            image.save(os.path.join(target_category_dir, img_name))  # 保存原图
            for idx, aug_img in enumerate(augmented_images):
                aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{idx + 1}.png"
                aug_img.save(os.path.join(target_category_dir, aug_img_name))

def dataset_augmentation(source_dir, target_dir):
    random.seed(42)  # 设置随机种子以保证结果可复现

    # 数据集路径
    source_dir = source_dir
    target_dir = target_dir

    # 类别
    categories = ['paper', 'rock', 'scissors']

    # 数据增强参数
    augmentations = 5  # 每张图片生成的增强图片数量

    # 对每个类别进行数据增强
    for category in categories:
        print(f"Processing category: {category}")
        process_category(category, source_dir, target_dir, augmentations=augmentations)

    print("Data augmentation completed!")


