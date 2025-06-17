import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
import torchvision.models as models
import random

class MahjongImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, class_names=None):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        # 用传入的 class_names，保证顺序和模型一致
        if class_names is None:
            raise ValueError("必须传入 class_names")
        self.class_names = class_names
        class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        for cls in self.class_names:
            cls_folder = os.path.join(img_dir, cls)
            if not os.path.isdir(cls_folder):
                continue
            for f in os.listdir(cls_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_folder, f)
                    try:
                        with Image.open(img_path) as im:
                            im.verify()
                        self.img_paths.append(img_path)
                        self.labels.append(class_to_idx[cls])
                    except (UnidentifiedImageError, OSError):
                        print(f"警告: 跳过损坏或无法识别的图片 {img_path}")
        # 打乱样本顺序
        combined = list(zip(self.img_paths, self.labels))
        if combined:
            random.shuffle(combined)
            self.img_paths[:], self.labels[:] = zip(*combined)
            self.img_paths = list(self.img_paths)
            self.labels = list(self.labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize((130, 90)),
    transforms.ToTensor(),
    ])
class_names = ['none', 'tiao_1', 'tiao_2', 'tiao_3', 'tiao_4', 'tiao_5', 'tiao_6', 'tiao_7', 'tiao_8', 'tiao_9', 'tong_1', 'tong_2', 'tong_3', 'tong_4', 'tong_5', 'tong_6', 'tong_7', 'tong_8', 'tong_9', 'wan_1', 'wan_2', 'wan_3', 'wan_4', 'wan_5', 'wan_6', 'wan_7', 'wan_8', 'wan_9', 'zi_bai', 'zi_bei', 'zi_dong', 'zi_fa', 'zi_nan', 'zi_xi', 'zi_zhong']

class resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.res = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.res.fc = nn.Linear(self.res.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.res(x)

def train_classifier(img_dir='./last_cards', batch_size=32, epochs=60):
    checkpoint_path = 'mj_classifier.pth'
    start_epoch = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet(num_classes=len(class_names)).to(device)

    # 断点续训：如果有权重则加载
    if os.path.exists(checkpoint_path):
        print(f"检测到已存在权重文件 '{checkpoint_path}'，将继续训练。")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    dataset = MahjongImageDataset(img_dir, transform=transform, class_names=class_names)
    if len(dataset) == 0:
        print("没有可用的训练样本，退出。")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save({'model': model.state_dict(), 'classes': class_names}, 'mj_classifier.pth')
    torch.save({'model': model.state_dict(), 'classes': class_names}, 'mj_classifier.pth')
    print("训练完成，模型已保存")
if __name__ == '__main__':
    train_classifier()
