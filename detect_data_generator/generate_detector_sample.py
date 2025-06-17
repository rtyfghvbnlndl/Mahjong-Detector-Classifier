import os
import random
import numpy as np
from PIL import Image, ImageDraw
import glob
import json

class MahjongSampleGenerator:
    def __init__(self):
        self.card_base_path = "empty_card.png"
        self.templates_dir = "./templates"
        self.output_dir = "./training_samples"
        self.annotations_file = os.path.join(self.output_dir, "annotations.json")
        self.template_size = (90, 130)  # 宽x高
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载底板
        self.card_base = Image.open(self.card_base_path).convert("RGBA")
        
        # 加载所有模板
        self.templates = self.load_templates()
        
        # 初始化标注数据
        self.annotations = {}
    
    def load_templates(self):
        """加载templates文件夹中的所有图案"""
        templates = []
        template_names = []
        pattern_files = glob.glob(os.path.join(self.templates_dir, "*"))
        
        for file_path in pattern_files:
            try:
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    template = Image.open(file_path).convert("RGBA")
                    # 调整模板尺寸到90x130
                    template = template.resize(self.template_size, Image.Resampling.LANCZOS)
                    templates.append(template)
                    # 记录模板名称（不含路径和扩展名）
                    template_name = os.path.splitext(os.path.basename(file_path))[0]
                    template_names.append(template_name)
            except Exception as e:
                print(f"无法加载模板 {file_path}: {e}")
        
        if not templates:
            raise ValueError("templates文件夹中没有找到有效的图案文件")
        
        self.template_names = template_names
        return templates
    
    def generate_random_background(self):
        """生成随机颜色和尺寸的背景"""
        # 随机尺寸 1200x900 ± 200
        width = random.randint(1000, 1400)
        height = random.randint(700, 1100)
        
        # 随机背景颜色
        bg_color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        
        background = Image.new("RGB", (width, height), bg_color)
        return background
    
    def find_black_region(self, card_base):
        """查找底板中的黑色区域位置"""
        # 将图片转换为numpy数组进行分析
        card_array = np.array(card_base.convert("RGB"))
        
        # 查找黑色像素（RGB值都很低）
        black_mask = np.all(card_array < 50, axis=2)
        
        if not np.any(black_mask):
            # 如果没找到黑色区域，返回默认位置
            return (0, 0)
        
        # 找到黑色区域的边界
        rows, cols = np.where(black_mask)
        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()
        
        return (left, top)
    
    def create_mahjong_card(self):
        """创建一张麻将牌（底板+随机图案）"""
        # 复制底板
        card = self.card_base.copy()
        
        # 随机选择一个图案
        template_index = random.randint(0, len(self.templates) - 1)
        template = self.templates[template_index]
        template_name = self.template_names[template_index]
        
        # 查找黑色区域位置
        black_pos = self.find_black_region(card)
        
        # 将图案贴到黑色区域上
        card.paste(template, black_pos, template if template.mode == 'RGBA' else None)
        
        return card, template_name
    
    def generate_sample(self, sample_id):
        """生成一个训练样本"""
        # 创建随机背景
        background = self.generate_random_background()
        bg_width, bg_height = background.size
        
        # 随机决定麻将牌数量 (8-15张)
        num_cards = random.randint(8, 15)
        
        # 获取麻将牌尺寸
        card_width, card_height = self.card_base.size
        
        # 记录已放置的麻将牌位置和信息
        placed_cards = []
        sample_annotations = []
        
        # 放置麻将牌
        for i in range(num_cards):
            # 创建麻将牌
            card, template_name = self.create_mahjong_card()
            
            # 尝试找到不重叠的位置
            max_attempts = 50
            placed = False
            
            for attempt in range(max_attempts):
                # 随机位置（确保不超出边界）
                max_x = bg_width - card_width
                max_y = bg_height - card_height
                
                if max_x > 0 and max_y > 0:
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    
                    # 检查是否与已放置的牌重叠
                    new_rect = (x, y, x + card_width, y + card_height)
                    overlaps = False
                    
                    for existing_rect in placed_cards:
                        if self.rectangles_overlap(new_rect, existing_rect[0]):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        # 将麻将牌贴到背景上
                        if card.mode == 'RGBA':
                            background.paste(card, (x, y), card)
                        else:
                            background.paste(card, (x, y))
                        
                        placed_cards.append((new_rect, template_name))
                        
                        # 记录标注信息
                        annotation = {
                            "x": x,
                            "y": y,
                            "width": card_width,
                            "height": card_height,
                            "template": template_name
                        }
                        sample_annotations.append(annotation)
                        placed = True
                        break
            
            if not placed:
                print(f"警告: 无法为第 {i+1} 张牌找到不重叠的位置")
        
        # 保存样本
        sample_filename = f"sample_{sample_id:04d}.jpg"
        output_path = os.path.join(self.output_dir, sample_filename)
        background.save(output_path, "JPEG", quality=95)
        
        # 保存标注信息
        self.annotations[sample_filename] = {
            "image_width": bg_width,
            "image_height": bg_height,
            "cards": sample_annotations
        }
        
        print(f"已生成样本: {output_path} (放置了 {len(sample_annotations)} 张牌)")
        
        return output_path
    
    def rectangles_overlap(self, rect1, rect2):
        """检查两个矩形是否重叠"""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        
        # 如果一个矩形在另一个的左侧、右侧、上方或下方，则不重叠
        if x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1:
            return False
        
        return True
    
    def generate_batch(self, num_samples=100):
        """批量生成训练样本，自动续编号，不覆盖已有数据"""
        print(f"准备生成 {num_samples} 个训练样本...")
        print(f"找到 {len(self.templates)} 个图案模板")

        # 读取已有标注文件
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                try:
                    self.annotations = json.load(f)
                except Exception:
                    self.annotations = {}
        else:
            self.annotations = {}

        # 读取已有图片编号
        existing_files = [fname for fname in os.listdir(self.output_dir) if fname.startswith("sample_") and fname.endswith(".jpg")]
        existing_ids = []
        for fname in existing_files:
            try:
                num = int(fname.replace("sample_", "").replace(".jpg", ""))
                existing_ids.append(num)
            except Exception:
                continue
        if self.annotations:
            for fname in self.annotations.keys():
                if fname.startswith("sample_") and fname.endswith(".jpg"):
                    try:
                        num = int(fname.replace("sample_", "").replace(".jpg", ""))
                        existing_ids.append(num)
                    except Exception:
                        continue

        existing_ids = sorted(set(existing_ids))
        if existing_ids:
            start_id = existing_ids[-1] + 1
        else:
            start_id = 1

        print(f"已有样本数量: {len(existing_ids)}，新样本编号从 {start_id} 开始")

        # 生成新样本
        for i in range(num_samples):
            sample_id = start_id + i
            try:
                self.generate_sample(sample_id)
            except Exception as e:
                print(f"生成样本 {sample_id} 时出错: {e}")

        # 保存合并后的标注文件
        with open(self.annotations_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)

        print(f"完成！样本保存在 {self.output_dir} 文件夹中")
        print(f"标注文件保存为: {self.annotations_file}")

def main():
    """主函数"""
    try:
        generator = MahjongSampleGenerator()
        
        # 生成100个样本（可以修改数量）
        generator.generate_batch(1000)
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请确保:")
        print("1. cards.png 文件存在")
        print("2. ./templates 文件夹存在且包含图案文件")

if __name__ == "__main__":
    main()
