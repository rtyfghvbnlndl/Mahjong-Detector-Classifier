import torch
import os
import random
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

from MahjongClassifier_train import MahjongImageDataset, resnet, transform, class_names

def test_classifier(img_dir='./classifier_train', num_samples=10):
    device = torch.device('cpu')
    checkpoint_path = 'mj_classifier.pth'
    if not os.path.exists(checkpoint_path):
        print("未找到训练好的权重文件 'mj_classifier.pth'")
        return

    # 加载模型
    checkpoint = torch.load('mj_classifier.pth', map_location=device)
    model = resnet(num_classes=len(class_names))
    print(class_names)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    # 加载数据集
    dataset = MahjongImageDataset(img_dir, transform=transform, class_names=class_names)
    if len(dataset) == 0:
        print("没有可用的测试样本，退出。")
        return
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    correct = 0
    cols = min(num_samples, 5)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
        img_path = dataset.img_paths[idx]
        true_label = class_names[label]
        pred_label = class_names[pred]
        if pred == label:
            correct += 1

        # 可视化并加框
        img_pil = Image.open(img_path)
        axes[i].imshow(img_pil)
        w, h = img_pil.size
        rect = plt.Rectangle((0, 0), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[i].add_patch(rect)
        axes[i].set_title(f"GT:{true_label}\nPred:{pred_label}", fontsize=10)
        axes[i].axis('off')

        # 控制台打印
        print(f"图片: {os.path.basename(img_path)} | 真实类别: {true_label} | 预测类别: {pred_label}")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    acc = correct / len(indices) if indices else 0
    plt.suptitle(f"分类准确率: {acc:.2f} ({correct}/{len(indices)})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default='./classifier_train', help='图片文件夹路径')
    args = parser.parse_args()

    test_classifier(img_dir=args.data_dir)