import os
import cv2

def process_image(image_path, size=(64, 64)):
    """
    处理单张图像：
    1. 转换为灰度图
    2. 应用 Sobel 算子提取边缘
    3. 计算梯度幅值
    4. 归一化到0-255
    5. 二值化处理，边缘为白色，其他部分为黑色
    6. 调整图像大小到指定尺寸
    返回处理后的单通道二值化图像
    """
    # 直接以灰度模式读取图像，确保是单通道
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 应用 Sobel 算子计算 x 和 y 方向的梯度
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    # 将梯度幅值归一化到 0-255
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_magnitude = gradient_magnitude.astype('uint8')

    # 二值化处理，阈值可以根据需要调整
    _, binary = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

    # 调整图像大小到指定尺寸
    resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)

    return resized

def create_target_directory(target_dir, class_name):
    """
    创建目标子目录，如果不存在则创建
    """
    target_class_dir = os.path.join(target_dir, class_name)
    os.makedirs(target_class_dir, exist_ok=True)
    return target_class_dir

def process_dataset(source_to_target_dirs, size=(64, 64)):
    """
    处理整个数据集：
    - source_to_target_dirs: 字典，键为源数据集目录，值为对应的目标基目录
    - size: 调整后的图像大小
    """
    classes = ['rock', 'paper', 'scissors']

    for source_dir, target_dir in source_to_target_dirs.items():
        if not os.path.isdir(source_dir):
            print(f"源目录不存在: {source_dir}")
            continue

        for class_name in classes:
            source_class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(source_class_dir):
                print(f"源子目录不存在: {source_class_dir}")
                continue

            # 创建对应的目标子目录
            target_class_dir = create_target_directory(target_dir, class_name)

            # 遍历源子目录中的所有图像文件
            for filename in os.listdir(source_class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    source_image_path = os.path.join(source_class_dir, filename)
                    processed_image = process_image(source_image_path, size=size)

                    if processed_image is not None:
                        # 定义目标图像路径
                        target_image_path = os.path.join(target_class_dir, filename)
                        # 保存处理后的图像
                        # 使用IMWRITE_GRAYSCALE参数确保图像保存为单通道
                        cv2.imwrite(target_image_path, processed_image)
                        print(f"已处理并保存: {target_image_path}")
                else:
                    print(f"跳过非图像文件: {filename}")

if __name__ == "__main__":
    # 定义源数据集目录和对应的目标基目录
    source_to_target = {
        'rps_binary': 'rps-binary64',
        'rps-test-set_binary': 'rps-test-set-binary64'
    }

    # 处理数据集
    process_dataset(source_to_target, size=(64, 64))

    print("所有图像处理完成。")
