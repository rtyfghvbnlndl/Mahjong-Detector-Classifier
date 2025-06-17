import torch
import torch.nn as nn
import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from MahjongDetector_train import YOLOMahjongDetector, MahjongDataset, transform

import argparse  # 新增

def test_model(data_dir='./training_samples', annotations_file='./training_samples/annotations.json', boxes_per_cell=3):
    # 设备选择
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 检查权重文件
    checkpoint_path = 'mahjong_detector.pth'
    if not os.path.exists(checkpoint_path):
        print("错误：未找到训练好的权重文件 'mahjong_detector.pth'")
        return
    
    # 加载模型，适配YOLO输出
    model = YOLOMahjongDetector(boxes_per_cell=boxes_per_cell).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("模型权重加载成功")
    except Exception as e:
        print("模型权重加载失败，模型结构可能已更改。")
        return
    model.eval()
    
    # 加载数据集
    if data_dir != None and annotations_file == None:
        annotations_file = data_dir + '/annotations.json'
    dataset = MahjongDataset(data_dir, annotations_file, transform=transform, boxes_per_cell=boxes_per_cell)
    
    # 随机选择10个样本进行测试
    test_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    total_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            image, label = dataset[idx]
            image_name = dataset.image_names[idx]
            
            # 预测
            image_batch = image.unsqueeze(0).to(device)
            det = model(image_batch)
            predictions = det[0].cpu()  # [S, S, N, 4]
            
            # 加载原始图像用于可视化
            original_image = Image.open(os.path.join(data_dir, image_name))  # 修改为data_dir
            
            # 获取真实标注
            annotation = dataset.annotations[image_name]
            img_width = annotation['image_width']
            img_height = annotation['image_height']
            
            # 可视化结果
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 真实标注
            ax1.imshow(original_image)
            ax1.set_title(f'Ground Truth - {image_name}')
            ax1.axis('off')
            
            for card in annotation['cards']:
                rect = patches.Rectangle(
                    (card['x'], card['y']), card['width'], card['height'],
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                ax1.add_patch(rect)
            
            # 预测结果
            ax2.imshow(original_image)
            ax2.set_title(f'Predictions (YOLO, conf>0.5)')
            ax2.axis('off')
            
            S = predictions.shape[0]
            N = predictions.shape[2]
            detected_cards = 0
            threshold = 0.5

            for y in range(S):
                for x in range(S):
                    for n in range(N):
                        conf = predictions[y, x, n, 0].item()
                        if conf > threshold:
                            rel_x1 = predictions[y, x, n, 1].item()
                            rel_y1 = predictions[y, x, n, 2].item()
                            rel_x2 = predictions[y, x, n, 3].item()
                            rel_y2 = predictions[y, x, n, 4].item()
                            # 还原到整图归一化坐标
                            x1 = (x + rel_x1) / S
                            y1 = (y + rel_y1) / S
                            x2 = (x + rel_x2) / S
                            y2 = (y + rel_y2) / S
                            # 还原到像素
                            box_x = (x2 + x1) /2* img_width
                            box_y = (y2 + y1)/2 * img_height
                            box_w = (x2 - x1) * img_width
                            box_h = (y2 - y1) * img_height
                            detected_cards += 1
                            rect = patches.Rectangle(
                                (box_x, box_y), box_w, box_h,
                                linewidth=2, edgecolor='red', facecolor='none'
                            )
                            ax2.add_patch(rect)
                            ax2.text(box_x, box_y-5, f'{conf:.2f}', color='red', fontsize=8, weight='bold')
            
            num_cards_int = len(annotation['cards'])
            plt.suptitle(f'Test {i+1}: GT={num_cards_int} cards, Detected={detected_cards} cards')
            plt.tight_layout()
            plt.show()
            
            # 计算简单准确率（检测数量准确性）
            accuracy = 1.0 - abs(num_cards_int - detected_cards) / max(num_cards_int, 1)
            total_accuracy += accuracy
            total_samples += 1
            
            print(f"样本 {i+1}: 真实={num_cards_int}, 预测={detected_cards}, 准确率={accuracy:.2f}")
    
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
    print(f"\n平均检测准确率: {avg_accuracy:.2f}")
    print("测试完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default=None, help='指定样本图片文件夹（位置参数）')
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
        annotations_file = os.path.join(data_dir, 'annotations.json')
    else:
        data_dir = './training_samples'
        annotations_file = './training_samples/annotations.json'
    test_model(data_dir=data_dir, annotations_file=annotations_file, boxes_per_cell=3)


