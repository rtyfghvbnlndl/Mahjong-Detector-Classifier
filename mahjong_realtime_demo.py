import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import threading

import torch
import torchvision
import sys
import os

# 保证可以import本地模块
sys.path.append(os.path.dirname(__file__))
from mahjong_detect_tool import detect_and_classify_mahjong

# 屏幕截图依赖
try:
    import mss
except ImportError:
    print("请先安装mss库: pip install mss")
    exit(1)

def grab_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if region is None else {"top": region[1], "left": region[0], "width": region[2], "height": region[3]}
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        return img

def show_realtime():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    img_disp = None

    while True:
        # 截屏
        img = grab_screen()
        # 直接传递PIL对象，不保存为文件
        count, boxes = detect_and_classify_mahjong(img)

        # 可视化
        ax.clear()
        ax.imshow(img)
        for box in boxes:
            rect = plt.Rectangle(
                (box['x'], box['y']),
                box['width'],
                box['height'],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            text_y = box['y'] - 10 if box['y'] - 10 > 0 else box['y'] + 15
            ax.text(
                box['x'],
                text_y,
                f"{box['class']} {box['score']:.2f} | {box['conf']:.2f}",
                color='yellow',
                fontsize=14,
                weight='bold',
                bbox=dict(facecolor='red', alpha=0.7, pad=2, edgecolor='none')
            )
        ax.set_title(f"实时麻将检测+分类（共{count}张）")
        ax.axis('off')
        plt.pause(1.0)

if __name__ == '__main__':
    show_realtime()


