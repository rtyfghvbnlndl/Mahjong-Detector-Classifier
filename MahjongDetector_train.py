import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models

import argparse  # 新增

SAM_NUM = 200  # 训练样本数量
EPO_NUM = 2000

class MahjongDataset(Dataset):
    def __init__(self, data_dir, annotations_file, max_cards=45, grid_size=13, boxes_per_cell=3, transform=None):
        self.data_dir = data_dir
        self.max_cards = max_cards
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell  # 新增
        self.transform = transform
        
        # 加载标注数据
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        # 只保留图片文件名（排除非图片项）
        self.image_names = [k for k in self.annotations.keys() if k.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        annotation = self.annotations[image_name]
        cards = annotation['cards']
        img_width = annotation['image_width']
        img_height = annotation['image_height']

        # YOLO标签: [S, S, N, 4] (置信度, x, y, h)
        S = self.grid_size
        N = self.boxes_per_cell
        # 修改为每格N个目标，每目标5个值（conf, x1, y1, x2, y2）
        label = torch.zeros((S, S, N, 5))
        assigned = np.zeros((S, S, N), dtype=bool)
        for card in cards[:self.max_cards]:
            # 归一化左上和右下坐标
            # cx, cy, w, h = card['x'], card['y'], card['width'], card['height']
            x1 = (card['x'] - card['width'] / 2) / img_width
            y1 = (card['y'] - card['height'] / 2) / img_height
            x2 = (card['x'] + card['width'] / 2) / img_width
            y2 = (card['y'] + card['height'] / 2) / img_height
            # 计算中心点决定分配到哪个格
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            grid_x = int(cx * S)
            grid_y = int(cy * S)
            grid_x = min(grid_x, S - 1)
            grid_y = min(grid_y, S - 1)
            for n in range(N):
                if not assigned[grid_y, grid_x, n]:
                    label[grid_y, grid_x, n, 0] = 1  # conf
                    # 相对该格左上角的归一化坐标
                    label[grid_y, grid_x, n, 1] = x1 * S - grid_x
                    label[grid_y, grid_x, n, 2] = y1 * S - grid_y
                    label[grid_y, grid_x, n, 3] = x2 * S - grid_x
                    label[grid_y, grid_x, n, 4] = y2 * S - grid_y
                    assigned[grid_y, grid_x, n] = True
                    break
        return image, label

class YOLOMahjongDetector(nn.Module):
    def __init__(self, boxes_per_cell=3):
        super().__init__()
        self.grid_size = (13, 13)
        self.boxes_per_cell = boxes_per_cell
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Conv2d(1280, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, boxes_per_cell * 5, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = nn.functional.interpolate(x, size=self.grid_size, mode='bilinear')
        B, C, S, S2 = x.shape
        N = self.boxes_per_cell
        # [B, N*5, S, S] -> [B, N, 5, S, S] -> [B, S, S, N, 5]
        x = x.view(B, N, 5, S, S2).permute(0, 3, 4, 1, 2)  # [B, S, S, N, 5]
        x[..., 0] = torch.sigmoid(x[..., 0])  # 只对conf用sigmoid
        # x[..., 1:5] 不加sigmoid
        return x

transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def yolo_loss(pred, target):
    # 标签和预测均为(conf, x1, y1, x2, y2)，x1/y1/x2/y2为归一化左上和右下点
    obj_mask = target[..., 0] > 0
    conf_loss = nn.BCELoss()(pred[..., 0], target[..., 0])
    if obj_mask.sum() > 0:
        # 坐标损失：直接对x1, y1, x2, y2做MSE
        coord_loss = nn.MSELoss()(pred[..., 1:5][obj_mask], target[..., 1:5][obj_mask])
    else:
        coord_loss = 0.0
    return conf_loss + 5.0 * coord_loss

def train_model(data_dir='./training_samples', annotations_file='./training_samples/annotations.json', boxes_per_cell=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    # 加载数据集
    full_dataset = MahjongDataset(data_dir, annotations_file, transform=transform, boxes_per_cell=boxes_per_cell)
    model = YOLOMahjongDetector(boxes_per_cell=boxes_per_cell).to(device)
    checkpoint_path = 'mahjong_detector.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("发现已保存的权重，正在加载...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("权重加载成功")
        except Exception as e:
            print("权重加载失败，模型结构可能已更改，需重新训练。")
        start_epoch = checkpoint.get('epoch', 0)
        print(f"从第 {start_epoch} 轮开始继续训练")
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    if os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    num_epochs = EPO_NUM
    total_epochs = start_epoch + num_epochs
    model.train()
    avg_loss = 0.0
    for epoch in range(start_epoch, total_epochs):
        if len(full_dataset) > SAM_NUM:
            indices = random.sample(range(len(full_dataset)), SAM_NUM)
            epoch_dataset = torch.utils.data.Subset(full_dataset, indices)
        else:
            epoch_dataset = full_dataset
        dataloader = DataLoader(epoch_dataset, batch_size=20, shuffle=True)
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = yolo_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:.4f}')
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"权重已保存到 {checkpoint_path}")
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print("训练完成，权重已保存")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default=None, help='指定样本图片文件夹（位置参数）')
    parser.add_argument('--boxes_per_cell', type=int, default=3, help='每格预测目标数')
    args = parser.parse_args()

    # 优先使用位置参数data_dir，否则默认
    if args.data_dir:
        data_dir = args.data_dir
        annotations_file = os.path.join(data_dir, 'annotations.json')
        train_model(data_dir=data_dir, annotations_file=annotations_file, boxes_per_cell=args.boxes_per_cell)
    else:
        data_dir = './training_samples'
        annotations_file = './training_samples/annotations.json'
        train_model(data_dir=data_dir, annotations_file=annotations_file, boxes_per_cell=args.boxes_per_cell)


