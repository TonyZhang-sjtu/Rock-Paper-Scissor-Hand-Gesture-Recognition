import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random
from aug_dataset import dataset_augmentation


# 设置随机种子函数
def set_seed(seed=42):
    """
    设置随机种子以确保实验可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True  # 确保每次卷积结果一致
    torch.backends.cudnn.benchmark = False


# 定义CNN模型
class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        # kernel_size=5的时候，调整padding为2来保证卷积之后大小不发生改变
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # 假设输入为224x224的图像
        self.dropout = nn.Dropout(p=0.5) # 初始dropout rate是0.5
        self.fc2 = nn.Linear(128, 3)  # 输出3类
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class GestureCNN_v1(nn.Module):
    def __init__(self):
        super(GestureCNN_v1, self).__init__()
        # kernel_size=5的时候，调整padding为2来保证卷积之后大小不发生改变
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # 假设输入为224x224的图像
        self.dropout = nn.Dropout(p=0.5) # 初始dropout rate是0.5
        self.fc2 = nn.Linear(128, 3)  # 输出3类
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
  

# 数据加载
def load_data(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到CNN输入的标准大小
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# 模型训练
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs=20, device="cpu"):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        test_acc = evaluate_model(model, test_loader, device="mps", confusion_mat=0)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs, test_accs


# 测试模型并绘制混淆矩阵
def evaluate_model(model, test_loader, device="cpu", confusion_mat=1):
    model.eval()
    all_preds, all_labels = [], []
    correct = 0  # 用来统计正确预测的样本数
    total = 0    # 用来统计总样本数
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 计算准确率
            correct += (predicted == labels).sum().item()  # 累加正确的个数
            total += labels.size(0)  # 累加总样本数
    
    # 计算准确率
    accuracy = correct / total * 100  # 准确率百分比
    
    # 输出准确率
    print(f"Test Accuracy: {accuracy:.2f}%")

    if confusion_mat:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rock", "Paper", "Scissors"])
        disp.plot(cmap="Blues")
        plt.show()

    return accuracy


# 绘制训练过程中的损失和准确率
def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    """
    绘制训练损失和准确率的折线图

    参数：
    - train_losses: list，训练损失值
    - val_losses: list，验证损失值
    - train_accs: list，训练准确率
    - val_accs: list，验证准确率
    """
    epochs = range(1, len(train_losses) + 1)  # 每个 epoch 的编号

    # 创建并排的两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 图 1：损失值变化
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Validation Loss")
    ax1.set_title("Loss vs. Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid()

    # 图 2：准确率变化
    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, val_accs, label="Validation Accuracy")
    ax2.set_title("Accuracy vs. Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid()

    # 显示图像
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 使用 argparse 来解析命令行参数
    parser = argparse.ArgumentParser(description="Train a CNN for Rock-Paper-Scissors Classification")
    parser.add_argument("--train_dir", type=str, default="./rps", help="Path to the training dataset directory")
    parser.add_argument("--test_dir", type=str, default="./rps-test-set", help="Path to the testing dataset directory")
    parser.add_argument("--source_dir", type=str, default="./rps", help="Path to the original dataset directory for augmentation")
    parser.add_argument("--target_dir", type=str, default="./rps_aug", help="Path to the augmented dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (e.g., cuda, mps, cpu)")
    parser.add_argument("--aug", type=int, default=0, help="1 need dataset augmentation; 0 unnecessary")
    parser.add_argument("--model", type=str, default="cnn", help="Choose the model to recognize gestures")
    args = parser.parse_args()

    set_seed(42)

    if args.aug:
        dataset_augmentation(args.source_dir, args.target_dir)
    
    # 选择设备
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():  # 检查是否支持 MPS
        device = torch.device("mps")
    elif torch.cuda.is_available():  # 检查是否支持 CUDA
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 加载数据
    train_loader, val_loader, test_loader = load_data(args.train_dir, args.test_dir, batch_size=args.batch_size)

    # 初始化模型
    if args.model == "cnn":
        model = GestureCNN().to(device)
    elif args.model == "cnn_v1":
        model = GestureCNN_v1().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练模型，并记录返回的结果
    train_losses, val_losses, train_accs, val_accs, test_accs = train_model(
        model, train_loader, val_loader, test_loader, criterion, optimizer, epochs=args.epochs, device=device
    )

    # 找到最大的test_accs的值，输出其值和下标+1作为最佳的训练epoch
    max_acc = max(test_accs)
    best_epoch = test_accs.index(max_acc) + 1
    print(f"Best Acc: {max_acc:.2f}%, Best Epoch: {best_epoch}")

    # 测试模型并绘制混淆矩阵
    evaluate_model(model, test_loader, device=device)

    # 绘制训练损失和准确率变化的图像
    plot_training_results(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()
