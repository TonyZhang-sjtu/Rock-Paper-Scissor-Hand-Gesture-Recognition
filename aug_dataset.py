import os
import random
from PIL import Image, ImageOps, ImageDraw
import numpy as np

# 定义数据增强函数
def augment_image(image, rotation_range=90, width_shift_range=0.2, height_shift_range=0.2,
                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'):
    """
    对图片进行数据增强，模拟ImageDataGenerator中的变换，并随机选择两种增强方式应用。

    参数：
    - image: PIL.Image 对象
    - rotation_range: 随机旋转的角度范围
    - width_shift_range: 水平平移范围
    - height_shift_range: 垂直平移范围
    - shear_range: 剪切强度
    - zoom_range: 缩放范围
    - horizontal_flip: 是否进行水平翻转
    - fill_mode: 填充模式（'nearest', 'constant', 'reflect', 'wrap'）
    
    返回：
    - augmented_images: 增强后的图片
    """
    enhanced_images = []  # 用于存储增强后的图片
    
    # 如果图片是 RGBA 模式，转换为 RGB
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # 定义可用的增强方式
    augmentations = []
    
    # 随机旋转
    augmentations.append(lambda img: img.rotate(random.randint(-rotation_range, rotation_range), 
                                                resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)))
    
    # 随机平移
    augmentations.append(lambda img: img.transform(img.size, Image.AFFINE, (1, 0, random.uniform(-width_shift_range, width_shift_range), 0, 1, random.uniform(-height_shift_range, height_shift_range)), 
                                                   resample=Image.BICUBIC, fillcolor=(255, 255, 255)))

    # 随机剪切
    augmentations.append(lambda img: img.transform(img.size, Image.AFFINE, (1, random.uniform(-shear_range, shear_range), 0, 0, 1, 0), 
                                                   resample=Image.BICUBIC, fillcolor=(255, 255, 255)))

    # 随机缩放
    augmentations.append(lambda img: img.resize((int(img.width * random.uniform(1 - zoom_range, 1 + zoom_range)),
                                                 int(img.height * random.uniform(1 - zoom_range, 1 + zoom_range))), 
                                                 resample=Image.BICUBIC))

    # 随机遮挡，遮挡范围是图片下半部分30%～40%区域，宽度是50%～100%
    if random.random() < 0.5:
        augmentations.append(lambda img: random_occlusion(img))

    # 随机水平翻转
    if horizontal_flip:
        augmentations.append(lambda img: ImageOps.mirror(img))

    # 填充模式
    def fill_image(img, mode=fill_mode):
        if mode == 'nearest':
            return img.convert("RGB")
        elif mode == 'constant':
            return img.convert("RGB", fillcolor=(255, 255, 255))
        elif mode == 'reflect':
            return ImageOps.expand(img, border=10, fill='reflect')
        elif mode == 'wrap':
            return ImageOps.expand(img, border=10, fill='wrap')

    # augmentations.append(lambda img: fill_image(img, fill_mode))


    # 随机选择两种增强方式进行应用
    random.shuffle(augmentations)  # 打乱顺序
    for augment in augmentations[:2]:  # 选择前两种
    # for augment in augmentations:  # 选择全部方法
        image_aug = augment(image.copy())
        enhanced_images.append(image_aug)
    
    return enhanced_images


def random_occlusion(image):
    """
    随机遮挡图片的下部，遮挡区域高度占整个图片30%-40%，宽度占50%-100%（居中）。
    
    参数：
    - image: PIL.Image 对象
    
    返回：
    - image: 被遮挡的图片
    """
    width, height = image.size
    occlusion_height = random.randint(int(0.3 * height), int(0.4 * height))  # 随机高度范围30%-40%
    occlusion_width = random.randint(int(0.5 * width), width)  # 随机宽度范围50%-100%
    
    # 定义遮挡区域的位置
    left = random.randint(0, int(width/2 - occlusion_width/2))  # 左上角X坐标
    upper = height - occlusion_height  # 下方区域，Y坐标
    right = left + occlusion_width  # 右上角X坐标
    lower = height  # 下方区域，Y坐标

    # 使用ImageDraw绘制遮挡区域
    draw = ImageDraw.Draw(image)
    draw.rectangle([left, upper, right, lower], fill=(255, 255, 255))  # 白色遮挡

    return image


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
    """
    对整个数据集进行数据增强。
    
    参数：
    - source_dir: str，原始数据集的根目录。
    - target_dir: str，增强后数据集的根目录。
    """
    random.seed(42)  # 设置随机种子以保证结果可复现

    # 类别
    categories = ['rock', 'paper', 'scissors']

    # 对每个类别进行数据增强
    for category in categories:
        print(f"Processing category: {category}")
        process_category(category, source_dir, target_dir, augmentations=5)

    print("Data augmentation completed!")
