import os
import random
import numpy as np
from PIL import Image, ImageDraw
import glob
import json

class NpEncoder(json.JSONEncoder):
    """解决numpy类型无法直接json序列化的问题"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class MahjongSampleGenerator:
    def __init__(self):
        self.card_base_path = "empty_card.png"
        self.templates_dir = "./templates"
        self.background_dir = "./background"  # 新增背景目录
        self.output_dir = "./training_samples_v3"  # 修改输出目录
        self.annotations_file = os.path.join(self.output_dir, "annotations.json")
        self.template_size = (90, 130)  # 宽x高

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载底板
        self.card_base = Image.open(self.card_base_path).convert("RGBA")

        # 加载所有模板
        self.templates = self.load_templates()

        # 加载所有背景图片路径
        self.background_paths = self.load_backgrounds()

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

    def load_backgrounds(self):
        """加载background文件夹中的所有图片路径"""
        pattern_files = glob.glob(os.path.join(self.background_dir, "*"))
        bg_paths = [f for f in pattern_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not bg_paths:
            raise ValueError("background文件夹中没有找到有效的背景图片")
        return bg_paths

    def generate_random_background(self):
        """从背景文件夹随机读取一张图片作为背景"""
        bg_path = random.choice(self.background_paths)
        background = Image.open(bg_path).convert("RGB")
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

    def get_effective_width(self, card_img):
        """获取麻将底板非透明区域的最左和最右像素坐标（用于无间隔排列）"""
        arr = np.array(card_img)
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            cols = np.any(alpha > 0, axis=0)
            nonzero_cols = np.where(cols)[0]
            if len(nonzero_cols) == 0:
                return 0, card_img.width - 1
            left = nonzero_cols[0]
            right = nonzero_cols[-1]
            return left, right
        else:
            # 没有alpha通道，直接用整张图
            return 0, card_img.width - 1

    def get_effective_box(self, card_img):
        """
        获取麻将底板非透明区域的最左、最右、最上、最下像素坐标（用于无间隔排列）
        """
        arr = np.array(card_img)
        if arr.shape[2] == 4:
            alpha = arr[..., 3]
            h, w = alpha.shape

            # 横向
            cols = np.any(alpha > 0, axis=0)
            nonzero_cols = np.where(cols)[0]
            if len(nonzero_cols) == 0:
                left, right = 0, w - 1
            else:
                left = nonzero_cols[0]
                right = nonzero_cols[-1]

            # 纵向
            rows = np.any(alpha > 0, axis=1)
            nonzero_rows = np.where(rows)[0]
            if len(nonzero_rows) == 0:
                top, bottom = 0, h - 1
            else:
                top = nonzero_rows[0]
                bottom = nonzero_rows[-1]

            return left, right, top, bottom
        else:
            return 0, card_img.width - 1, 0, card_img.height - 1

    def create_mahjong_card(self, target_width):
        """创建一张麻将牌（底板+随机图案），并缩放到指定宽度"""
        card = self.card_base.copy()
        template_index = random.randint(0, len(self.templates) - 1)
        template = self.templates[template_index]
        template_name = self.template_names[template_index]
        black_pos = self.find_black_region(card)
        card.paste(template, black_pos, template if template.mode == 'RGBA' else None)
        orig_w, orig_h = card.size
        scale = target_width / orig_w
        target_height = int(orig_h * scale)
        card = card.resize((int(target_width), target_height), Image.Resampling.LANCZOS)
        return card, template_name

    def generate_sample(self, sample_id):
        """生成一个训练样本"""
        background = self.generate_random_background()
        bg_width, bg_height = background.size

        # 随机麻将牌宽度为背景宽度的2%-6%
        card_width = int(bg_width * random.uniform(0.03, 0.06))
        num_templates = len(self.templates)
        # 先生成一张牌，获取有效区域
        temp_card, _ = self.create_mahjong_card(card_width)
        left, right, top, bottom = self.get_effective_box(temp_card)
        effective_width = right - left + 1
        effective_height = bottom - top + 1

        # 随机排数 1-3
        max_rows = min((bg_height - 1) // effective_height, 3)
        if max_rows < 1:
            max_rows = 1
        num_rows = random.randint(1, max_rows)

        sample_annotations = []
        used_templates = set()
        max_total_cards = min(num_templates, num_rows * ((bg_width - 1) // effective_width), 25)  # 限制最大45张
        if max_total_cards < 1:
            max_total_cards = 1

        # 随机纵向起始位置，保证最后一排不超出
        max_y = bg_height - (num_rows * effective_height)
        y0 = random.randint(0, max(0, max_y))

        template_indices_all = random.sample(range(num_templates), max_total_cards)
        template_idx = 0

        for row in range(num_rows):
            # 本排最多能放多少张
            max_cards_this_row = min((bg_width - 1) // effective_width, max_total_cards - template_idx)
            if max_cards_this_row < 1:
                break
            num_cards = random.randint(1, max_cards_this_row)
            # 保证总数不超过45
            if template_idx + num_cards > 45:
                num_cards = 45 - template_idx
            if num_cards <= 0:
                break

            # 随机横向起始位置，保证最后一张不超出
            max_x = bg_width - (num_cards * effective_width)
            x0 = random.randint(0, max(0, max_x))

            # 随机决定本排是否对齐（True为对齐，False为不对齐）
            align = random.choice([True, False])
            if align:
                x_offsets = [x0 + i * effective_width for i in range(num_cards)]
            else:
                possible_xs = list(range(0, bg_width - effective_width + 1, effective_width))
                if len(possible_xs) < num_cards:
                    x_offsets = [x0 + i * effective_width for i in range(num_cards)]
                else:
                    x_offsets = sorted(random.sample(possible_xs, num_cards))

            y = y0 + row * effective_height

            for i in range(num_cards):
                template_index = template_indices_all[template_idx]
                template_idx += 1
                card = self.card_base.copy()
                template = self.templates[template_index]
                template_name = self.template_names[template_index]
                black_pos = self.find_black_region(card)
                card.paste(template, black_pos, template if template.mode == 'RGBA' else None)
                orig_w, orig_h = card.size
                scale = card_width / orig_w
                target_height = int(orig_h * scale)
                card = card.resize((int(card_width), target_height), Image.Resampling.LANCZOS)
                left, right, top, bottom = self.get_effective_box(card)
                effective_width = right - left + 1
                effective_height = bottom - top + 1
                card_height = card.size[1]

                x = x_offsets[i]
                card_crop = card.crop((left, top, right + 1, bottom + 1))
                if card_crop.mode == 'RGBA':
                    background.paste(card_crop, (x, y), card_crop)
                else:
                    background.paste(card_crop, (x, y))
                annotation = {
                    "x": x,
                    "y": y,
                    "width": effective_width,
                    "height": effective_height,
                    "template": template_name
                }
                sample_annotations.append(annotation)
            if template_idx >= 45:
                break

        sample_filename = f"sample_{sample_id:04d}.jpg"
        output_path = os.path.join(self.output_dir, sample_filename)
        background.save(output_path, "JPEG", quality=95)

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
            json.dump(self.annotations, f, ensure_ascii=False, indent=2, cls=NpEncoder)
        print(f"完成！样本保存在 {self.output_dir} 文件夹中")
        print(f"标注文件保存为: {self.annotations_file}")

def main():
    """主函数"""
    try:
        generator = MahjongSampleGenerator()

        # 生成100个样本（可以修改数量）
        generator.generate_batch(2000)

    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请确保:")
        print("1. cards.png 文件存在")
        print("2. ./templates 文件夹存在且包含图案文件")



if __name__ == "__main__":
    main()
