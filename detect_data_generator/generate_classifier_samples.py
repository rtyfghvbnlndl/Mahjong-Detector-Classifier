import os
import random
from PIL import Image, ImageDraw

TEMPLATE_DIR = './templates'
BACKGROUND_DIR = './background'
TRAIN_DIR = './classifier_train'
LABEL_FILE = os.path.join(TRAIN_DIR, 'labels.txt')
IMG_W, IMG_H = 90, 130
SINGLE_PER_CLASS = 10  # 每类裁剪样本数
MIXED_PER_CLASS = 10   # 每类混合样本数
ORIGINAL_PER_CLASS = 5  # 每类原图样本数

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_templates():
    files = [f for f in os.listdir(TEMPLATE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    class2imgs = {}
    for f in files:
        cls = os.path.splitext(f)[0]
        img = Image.open(os.path.join(TEMPLATE_DIR, f)).convert('RGB')
        class2imgs.setdefault(cls, []).append(img)
    return class2imgs

def load_backgrounds():
    files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    bg_imgs = []
    for f in files:
        img = Image.open(os.path.join(BACKGROUND_DIR, f)).convert('RGB')
        bg_imgs.append(img)
    return bg_imgs

def random_bg_crop(bg_imgs):
    bg = random.choice(bg_imgs)
    if bg.width < IMG_W or bg.height < IMG_H:
        bg = bg.resize((max(bg.width, IMG_W), max(bg.height, IMG_H)))
    left = random.randint(0, bg.width - IMG_W)
    top = random.randint(0, bg.height - IMG_H)
    return bg.crop((left, top, left + IMG_W, top + IMG_H))

def crop_and_pad(img, bg_imgs):
    mode = random.choice(['left', 'right', 'both'])
    if mode == 'left':
        crop_ratio = random.uniform(0.1, 0.2)
        crop_px = int(IMG_W * crop_ratio)
        box = (crop_px, 0, IMG_W, IMG_H)
        cropped = img.crop(box)
        bg_img = random_bg_crop(bg_imgs)
        bg_img.paste(cropped, (0, 0))
        new_img = bg_img
    elif mode == 'right':
        crop_ratio = random.uniform(0.1, 0.2)
        crop_px = int(IMG_W * crop_ratio)
        box = (0, 0, IMG_W - crop_px, IMG_H)
        cropped = img.crop(box)
        bg_img = random_bg_crop(bg_imgs)
        bg_img.paste(cropped, (crop_px, 0))
        new_img = bg_img
    else:  # both sides
        total_crop_ratio = random.uniform(0.1, 0.2)
        left_ratio = random.uniform(0.05, total_crop_ratio - 0.05)
        right_ratio = total_crop_ratio - left_ratio
        left_px = int(IMG_W * left_ratio)
        right_px = int(IMG_W * right_ratio)
        box = (left_px, 0, IMG_W - right_px, IMG_H)
        cropped = img.crop(box)
        bg_img = random_bg_crop(bg_imgs)
        bg_img.paste(cropped, (left_px, 0))
        new_img = bg_img
    return new_img

def mix_images(img1, img2, cls1, cls2, bg_imgs):
    # mix时不能缩放，只能裁剪，且裁剪方向要与拼接方向一致，拼接后整体居中贴到背景上
    ratio1 = random.uniform(0.8, 0.9)
    w1 = int(IMG_W * ratio1)
    w2 = IMG_W - w1

    # 保证主图和副图高度为IMG_H，若原图高不够则补白
    def pad_to_height(img, target_h):
        if img.height == target_h:
            return img
        new_img = Image.new('RGB', (img.width, target_h), (255, 255, 255))
        top = (target_h - img.height) // 2
        new_img.paste(img, (0, top))
        return new_img

    img1 = pad_to_height(img1, IMG_H)
    img2 = pad_to_height(img2, IMG_H)

    # 主图裁剪（居中裁剪或随机裁剪）
    if img1.width > w1:
        left1 = random.randint(0, img1.width - w1)
        img1_cropped = img1.crop((left1, 0, left1 + w1, IMG_H))
    else:
        # 不够宽则补白
        img1_cropped = Image.new('RGB', (w1, IMG_H), (255, 255, 255))
        img1_cropped.paste(img1, ((w1 - img1.width)//2, 0))

    # 副图裁剪，方向与拼接方向一致
    if w1 > w2:
        # 主图在左，副图在右，副图裁左侧
        if img2.width > w2:
            img2_cropped = img2.crop((img2.width - w2, 0, img2.width, IMG_H))
        else:
            img2_cropped = Image.new('RGB', (w2, IMG_H), (255, 255, 255))
            img2_cropped.paste(img2, (w2 - img2.width, 0))
    else:
        # 主图在右，副图在左，副图裁右侧
        if img2.width > w2:
            img2_cropped = img2.crop((0, 0, w2, IMG_H))
        else:
            img2_cropped = Image.new('RGB', (w2, IMG_H), (255, 255, 255))
            img2_cropped.paste(img2, (0, 0))

    # 拼接
    mix_img = Image.new('RGB', (w1 + w2, IMG_H), (255, 255, 255))
    if w1 > w2:
        mix_img.paste(img1_cropped, (0, 0))
        mix_img.paste(img2_cropped, (w1, 0))
    else:
        mix_img.paste(img2_cropped, (0, 0))
        mix_img.paste(img1_cropped, (w2, 0))

    # 居中贴到背景
    bg_img = random_bg_crop(bg_imgs)
    x_offset = (IMG_W - mix_img.width) // 2
    y_offset = 0
    bg_img.paste(mix_img, (x_offset, y_offset))

    label = cls1
    return bg_img, label

def add_random_border(img):
    thickness = random.randint(5, 15)
    color = tuple(random.randint(0, 255) for _ in range(3))
    new_w = img.width + 2 * thickness
    new_h = img.height + 2 * thickness
    bordered = Image.new('RGB', (new_w, new_h), color)
    bordered.paste(img, (thickness, thickness))
    # 可选：再画一圈线条
    draw = ImageDraw.Draw(bordered)
    for t in range(thickness):
        draw.rectangle([t, t, new_w-1-t, new_h-1-t], outline=color)
    return bordered

def main():
    ensure_dir(TRAIN_DIR)
    class2imgs = load_templates()
    class_names = list(class2imgs.keys())
    bg_imgs = load_backgrounds()
    label_lines = []

    # 新增：原图样本
    for cls in class_names:
        imgs = class2imgs[cls]
        for i in range(ORIGINAL_PER_CLASS):
            img = random.choice(imgs).resize((IMG_W, IMG_H))
            img = add_random_border(img)
            fname = f'{cls}_orig_{i}.png'
            out_path = os.path.join(TRAIN_DIR, fname)
            img.save(out_path)
            label_lines.append(f'{fname},{cls}\n')

    # 单图裁剪样本
    for cls in class_names:
        imgs = class2imgs[cls]
        for i in range(SINGLE_PER_CLASS):
            img = random.choice(imgs).resize((IMG_W, IMG_H))
            out_img = crop_and_pad(img, bg_imgs)
            out_img = add_random_border(out_img)
            fname = f'{cls}_crop_{i}.png'
            out_path = os.path.join(TRAIN_DIR, fname)
            out_img.save(out_path)
            label_lines.append(f'{fname},{cls}\n')

    # 拼接样本（每个类别 MIXED_PER_CLASS 张）
    for cls1 in class_names:
        for i in range(MIXED_PER_CLASS):
            # 随机选一个不同类别
            cls2 = random.choice([c for c in class_names if c != cls1])
            img1 = random.choice(class2imgs[cls1]).resize((IMG_W, IMG_H))
            img2 = random.choice(class2imgs[cls2]).resize((IMG_W, IMG_H))
            out_img, label = mix_images(img1, img2, cls1, cls2, bg_imgs)
            out_img = add_random_border(out_img)
            fname = f'{label}_mix_{cls1}_{i}.png'
            out_path = os.path.join(TRAIN_DIR, fname)
            out_img.save(out_path)
            label_lines.append(f'{fname},{label}\n')

    # 保存标签
    with open(LABEL_FILE, 'w', encoding='utf-8') as f:
        f.writelines(label_lines)
    print(f"样本生成完毕，图片和标签已保存到 {TRAIN_DIR}")

if __name__ == '__main__':
    main()
