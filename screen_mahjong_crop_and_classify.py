import os
import time
from PIL import Image
import mss
import keyboard

from mahjong_detect_tool import detect_and_classify_mahjong

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 全屏
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        return img

def main():
    save_dir = './last_cards'
    ensure_dir(save_dir)
    print("按下 F8 截取屏幕并识别麻将牌，按 ESC 退出。")
    while True:
        if keyboard.is_pressed('esc'):
            print("退出。")
            break
        if keyboard.is_pressed('f8'):
            print("正在截屏...")
            img = screenshot()
            count, boxes = detect_and_classify_mahjong(img)
            print(f"检测到{count}张麻将牌，正在保存...")
            class_count = {}
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                crop = img.crop((x1, y1, x2, y2))
                cls = box['class']
                class_count.setdefault(cls, 0)
                class_count[cls] += 1
                fname = f"{cls}_{class_count[cls]}.png"
                crop.save(os.path.join(save_dir, fname))
                print(f"保存: {fname}")
            print("全部保存完成。等待下一次F8...")
            # 防止多次触发
            while keyboard.is_pressed('f8'):
                time.sleep(0.1)
        time.sleep(0.05)

if __name__ == '__main__':
    main()
