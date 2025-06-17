import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from MahjongDetector_train import YOLOMahjongDetector, transform as detector_transform
from MahjongClassifier_train import resnet, transform as classifier_transform, class_names as classifier_class_names

def detect_mahjong(image, model_weights='mahjong_detector.pth', conf_threshold=0.5, boxes_per_cell=None):
    """
    输入图片路径或PIL.Image对象，输出麻将牌的位置和数量
    返回: (数量, [每个麻将的box字典])
    box字典: {'x': 左上角x, 'y': 左上角y, 'width': 宽, 'height': 高, 'conf': 置信度}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载权重文件
    checkpoint = torch.load(model_weights, map_location=device)
    # 自动推断boxes_per_cell
    if boxes_per_cell is None:
        if 'model_state_dict' in checkpoint:
            head_weight = checkpoint['model_state_dict']['head.2.weight']
        else:
            head_weight = checkpoint['head.2.weight']
        out_channels = head_weight.shape[0]
        boxes_per_cell = out_channels // 5  # 5而不是4
    # 加载模型
    model = YOLOMahjongDetector(boxes_per_cell=boxes_per_cell).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    model.eval()

    # 预处理
    transform = detector_transform
    # 支持路径或PIL.Image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
    img_width, img_height = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        det = model(input_tensor)
        output = det[0].cpu()         # [S, S2, N, 5]

    S = output.shape[0]   # 高
    S2 = output.shape[1]  # 宽
    N = output.shape[2]
    results = []
    for y in range(S):
        for x in range(S2):
            for n in range(N):
                conf = output[y, x, n, 0].item()
                if conf > conf_threshold:
                    # 解析数据，这里坐标点需要通过abcd二次计算
                    rel_a = output[y, x, n, 1].item()
                    rel_b = output[y, x, n, 2].item()
                    rel_c = output[y, x, n, 3].item()
                    rel_d = output[y, x, n, 4].item()
                    
                    a = (x + rel_a) / S2
                    b = (y + rel_b) / S
                    c = (x + rel_c) / S2
                    d = (y + rel_d) / S
                    
                    results.append({
                        'x1': (a+c)/2,
                        'y1': (b+d)/2,
                        'x2': (a+c)/2 + c-a,
                        'y2': (b+d)/2 + d-b,
                        'conf': conf
                    })
    return len(results), results

def binarize(img, threshold=0.5):
    # img 是Tensor，像素范围[0,1]
    return (img > threshold).float()

# 分类模型加载与推理
def load_classifier(model_path='mj_classifier.pth', device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = classifier_class_names
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model = resnet(num_classes=len(class_names)).to(device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    transform = classifier_transform
    return model, class_names, transform

def detect_and_classify_mahjong(image, detector_weights='mahjong_detector.pth', classifier_weights='mj_classifier.pth',
                               grid_size=13, conf_threshold=0.5, boxes_per_cell=None):
    """
    输入图片路径或PIL.Image对象，输出每张麻将的位置（左上、右下）、分类和数量
    返回: (数量, [每个麻将的box字典（左上、右下），含class和score])
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 检测麻将位置
    count, det_boxes = detect_mahjong(
        image, model_weights=detector_weights, conf_threshold=conf_threshold, boxes_per_cell=boxes_per_cell
    )

    if count == 0:
        return 0, []

    # 2. 加载分类器
    classifier, class_names, classifier_transform = load_classifier(classifier_weights, device=device)

    # 3. 读取原始图片
    if isinstance(image, str):
        pil_img = Image.open(image).convert('RGB')
    else:
        pil_img = image.convert('RGB')
    img_width, img_height = pil_img.size

    results = []
    for box in det_boxes:
        # 坐标归一化转像素
        x1 = int(box['x1'] * img_width)
        y1 = int(box['y1'] * img_height)
        x2 = int(box['x2'] * img_width)
        y2 = int(box['y2'] * img_height)
        # 防止越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # 裁剪麻将牌区域
        crop = pil_img.crop((x1, y1, x2, y2))
        crop_tensor = classifier_transform(crop).unsqueeze(0).to(device)

        # # ====== 新增：显示crop_tensor为图片 ======
        # import matplotlib.pyplot as plt
        # import numpy as np
        # # 反归一化并转为numpy
        # show_tensor = crop_tensor.detach().cpu().squeeze(0)
        # if show_tensor.shape[0] == 3:
        #     show_tensor = show_tensor.permute(1, 2, 0)
        # show_img = show_tensor.numpy()
        # # 若有归一化，可在此反归一化（如mean/std），否则直接clip
        # show_img = np.clip(show_img, 0, 1)
        # plt.figure()
        # plt.imshow(show_img)
        # plt.title("Debug: crop_tensor as image")
        # plt.axis('off')
        # plt.show()
        # # ====== 新增结束 ======

        # 分类
        with torch.no_grad():
            logits = classifier(crop_tensor)
            prob = torch.softmax(logits, dim=1)
            score, pred = torch.max(prob, dim=1)
            pred_class = class_names[pred.item()]
            pred_score = score.item()
        # 合并结果
        results.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'conf': box['conf'],
            'class': pred_class,
            'score': pred_score
        })

    return len(results), results

# 示例用法
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_path = "test1.png"  # 替换为你的图片路径

    # 检测+分类
    count, boxes = detect_and_classify_mahjong(img_path)
    print(f"检测到麻将数量: {count}")
    for i, box in enumerate(boxes):
        print(f"麻将{i+1}: (x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}, conf={box['conf']}, class={box['class']}, score={box['score']:.2f})")

    # 合并可视化：检测框+分类结果
    image = Image.open(img_path).convert('RGB')
    img_width, img_height = image.size
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()
    for i, box in enumerate(boxes):
        # 坐标归一化转像素
        x1 = box['x1'] 
        y1 = box['y1'] 
        x2 = box['x2']
        y2 = box['y2'] 
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        text_y = y1 - 10 if y1 - 10 > 0 else y1 + 15
        ax.text(
            x1,
            text_y,
            f"{box['class']} {box['score']:.2f} | {box['conf']:.2f}",
            color='yellow',
            fontsize=8,
            weight='bold',
            bbox=dict(facecolor='red', alpha=0.7, pad=2, edgecolor='none')
        )
    plt.axis('off')
    plt.title(f"检测+分类结果（共{count}张）")
    plt.tight_layout()
    plt.show()