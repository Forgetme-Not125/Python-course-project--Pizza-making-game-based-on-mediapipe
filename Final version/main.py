# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import os
from enum import Enum
from PIL import Image, ImageDraw, ImageFont


# ==========================================
#         通用工具函数 (Utils)
# ==========================================
def draw_text_with_emoji(img, text, position, font_scale=0.8, color=(255, 255, 255), thickness=2):
    """使用 PIL 绘制支持表情符号的文字"""
    try:
        # 将 OpenCV 图像转换为 PIL 图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 尝试使用系统字体，如果失败则使用默认字体
        try:
            # Windows 系统字体（微软雅黑）
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(font_scale * 40))
        except:
            try:
                # 备用字体
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", int(font_scale * 40))
            except:
                # 使用默认字体
                font = ImageFont.load_default()

        # 绘制文字
        draw.text(position, text, font=font, fill=color)

        # 转换回 OpenCV 格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 如果 PIL 绘制失败，使用 OpenCV 绘制（不含表情符号）
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return img


def load_sprite(img_path, target_long=None, target_size=None):
    """安全加载图片，如果不存在返回 None"""
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    h, w = img.shape[:2]
    if target_size:
        img = cv2.resize(img, target_size)
    elif target_long:
        scale = target_long / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return img


def paste_sprite(bg, sprite, xy, center=False):
    """将带透明通道的 sprite 贴到 bg 上"""
    if sprite is None: return
    h, w = bg.shape[:2]
    sh, sw = sprite.shape[:2]
    x, y = xy

    if center:
        x = int(x - sw / 2)
        y = int(y - sh / 2)

    x, y = int(x), int(y)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + sw), min(h, y + sh)

    if x2 <= x1 or y2 <= y1: return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    roi = bg[y1:y2, x1:x2]
    spr = sprite[sy1:sy2, sx1:sx2]

    if spr.shape[2] == 4:
        alpha = spr[:, :, 3:4] / 255.0
        roi[:] = (1 - alpha) * roi + alpha * spr[:, :, :3]
    else:
        roi[:] = spr


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10):
    """绘制圆角矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness > 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def draw_progress_bar(img, x, y, width, height, progress, color=(0, 255, 0), bg_color=(50, 50, 50), text=""):
    """绘制美观的进度条"""
    draw_rounded_rect(img, (x, y), (x + width, y + height), bg_color, -1, radius=height // 2)
    fill_width = int(width * min(progress, 1.0))
    if fill_width > 0:
        draw_rounded_rect(img, (x, y), (x + fill_width, y + height), color, -1, radius=height // 2)
    draw_rounded_rect(img, (x, y), (x + width, y + height), (255, 255, 255), 2, radius=height // 2)

    if text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x + width // 2 - text_size[0] // 2
        text_y = y + height // 2 + text_size[1] // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_glowing_text(img, text, position, font_scale, color, thickness=2):
    """绘制带发光效果的文字"""
    x, y = position
    for i in range(3):
        alpha = 50 - i * 15
        overlay = img.copy()
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness + i * 2)
        cv2.addWeighted(overlay, alpha / 100.0, img, 1 - alpha / 100.0, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_styled_text(img, text, position, font_scale, color, thickness=2, bg_color=None):
    """绘制带背景样式的文字"""
    x, y = position

    # 简化实现，只显示文字，移除多余的装饰
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_gradient_rect(img, pt1, pt2, color1, color2):
    """绘制渐变矩形"""
    x1, y1 = pt1
    x2, y2 = pt2
    height = y2 - y1
    width = x2 - x1

    for i in range(height):
        alpha = i / height
        color = tuple(int(color1[j] * (1 - alpha) + color2[j] * alpha) for j in range(3))
        cv2.line(img, (x1, y1 + i), (x2, y1 + i), color, 1)


# ==========================================
#      第一关：擀面 (Dough Simulator)
# ==========================================
class PizzaState(Enum):
    WAITING = 1
    OVER_DOUGH = 2
    PRESSING = 3
    FINISHED = 4


class PizzaDoughSimulator:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.state = PizzaState.WAITING

        self.pizza_radius = 200
        self.pizza_center = (w // 2, h // 2)
        self.press_count = 0
        self.required_presses = 8
        self.progress = 0.0
        self.dough_thickness = 1.0

        self.last_palm_y = 0
        self.press_start_y = 0
        self.is_pressing_down = False
        self.last_press_time = 0

        self.message = "Please open your palm and move it above the dough."
        self.message_timer = 0
        self.last_quality_msg = ""
        self.last_quality_color = (100, 100, 100)

        self.dough_img = load_sprite('dough.jpg', target_size=(w, h))
        self.crust_img = load_sprite('crust.jpg', target_size=(w, h))

        if self.dough_img is None:
            self.dough_img = np.full((h, w, 3), (150, 150, 150), dtype=np.uint8)
            cv2.circle(self.dough_img, self.pizza_center, 150, (200, 190, 160), -1)

        self.current_img = self.dough_img.copy()

        # 评分系统
        self.press_qualities = []  # 记录每次按压的质量
        self.score = 0
        self.quality_grade = ""

    def set_message(self, text, duration=45):
        self.message = text
        self.message_timer = duration

    def process(self, frame, hand_landmarks):
        display = frame.copy()

        if self.current_img is not None:
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            display[mask == 255] = self.current_img[mask == 255]

        instructions = [
            "Instructions:",
            "1. Open hand, move to center",
            "2. Wave hand UP/DOWN to press",
            f"3. Need {self.required_presses} strong presses",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(display, line, (self.w - 450, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.message:
            # 去掉表情符号
            message_text = self.message.split(' ')[0] if ' ' in self.message and any(
                char in '💯👍' for char in self.message) else self.message
            text_size = cv2.getTextSize(message_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (self.w - text_size[0]) // 2
            text_y = self.h - 80  # 移动到屏幕下方
            overlay = display.copy()
            cv2.rectangle(overlay, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            cv2.putText(display, message_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if self.message_timer > 0:
                self.message_timer -= 1
            elif self.message_timer == 0 and self.state == PizzaState.PRESSING:
                self.message = "Swing palm UP and DOWN!"

        draw_styled_text(display, f"Phase 1: Dough ({self.press_count}/{self.required_presses})", (30, 50), 1,
                         (0, 255, 255), 2, (0, 0, 0))

        draw_progress_bar(display, 30, 70, 300, 20, self.progress, (0, 255, 0), (50, 50, 50),
                          f"{int(self.progress * 100)}%")

        if self.last_quality_msg:
            cv2.putText(display, f"{self.last_quality_msg}", (50, self.h - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        self.last_quality_color, 3)  # 增大字体和加粗

        if not hand_landmarks: return display

        lm = hand_landmarks[0].landmark
        wrist = lm[0]
        palm_x, palm_y = int(wrist.x * self.w), int(wrist.y * self.h)
        dist_finger = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        is_open = dist_finger > 0.05

        color = (0, 255, 0) if is_open else (0, 0, 255)
        cv2.circle(display, (palm_x, palm_y), 15, color, -1)
        cv2.line(display, (palm_x, palm_y), self.pizza_center, (200, 200, 200), 1)
        dist_center = math.hypot(palm_x - self.pizza_center[0], palm_y - self.pizza_center[1])

        if self.state == PizzaState.WAITING:
            cv2.circle(display, self.pizza_center, 100, (255, 255, 0), 2)
            if not is_open:
                self.set_message("Please OPEN your palm!", 5)
            elif dist_center > 180:
                self.set_message("Move palm to the DOUGH CENTER", 5)
            else:
                self.state = PizzaState.OVER_DOUGH
                self.set_message("Ready! Start swinging UP and DOWN", 60)

        elif self.state == PizzaState.OVER_DOUGH:
            self.last_palm_y = palm_y
            self.state = PizzaState.PRESSING
            self.set_message("Start pressing! Move hand UP/DOWN", 60)

        elif self.state == PizzaState.PRESSING:
            if not is_open:
                self.state = PizzaState.WAITING
                self.set_message("Palm closed. Please OPEN palm.", 45)
                return display
            if dist_center > 250:
                self.state = PizzaState.WAITING
                self.set_message("Palm has left the dough!", 45)
                return display

            dy = palm_y - self.last_palm_y
            if dy > 5 and not self.is_pressing_down:
                self.is_pressing_down = True
                self.press_start_y = self.last_palm_y

            if dy < -5 and self.is_pressing_down:
                press_depth = abs(palm_y - self.press_start_y)
                if press_depth > 40 and (time.time() - self.last_press_time > 0.5):
                    strength = min(1.0, press_depth / 150.0)
                    self.press_count += 1
                    self.last_press_time = time.time()
                    self.is_pressing_down = False
                    self.progress = self.press_count / self.required_presses
                    self.dough_thickness -= 0.1
                    self.update_dough_visual()

                    # 记录按压质量并计算分数
                    if strength > 0.8:  # 提高perfect的门槛
                        quality = "Perfect"
                        quality_score = 100
                        self.last_quality_color = (0, 255, 0)
                    elif strength > 0.5:  # 调整good的门槛
                        quality = "Good"
                        quality_score = 80  # 提高good的分数
                        self.last_quality_color = (0, 255, 255)
                    else:
                        quality = "OK"
                        quality_score = 60  # 提高OK的分数
                        self.last_quality_color = (0, 165, 255)

                    self.press_qualities.append(quality_score)
                    self.last_quality_msg = quality

                elif press_depth <= 40 and self.is_pressing_down:
                    self.set_message("!!! Swing amplitude not sufficient !!!", 30)
                    self.last_quality_msg = "Too weak"
                    self.last_quality_color = (0, 0, 255)
            self.last_palm_y = palm_y

            if self.press_count >= self.required_presses:
                self.calculate_final_score()
                self.state = PizzaState.FINISHED
                self.set_message("Rolling Complete! Next Phase...", 100)

        return display

    def update_dough_visual(self):
        if self.dough_img is None or self.crust_img is None: return
        alpha = min(1.0, self.progress)
        beta = 1.0 - alpha
        self.current_img = cv2.addWeighted(self.dough_img, beta, self.crust_img, alpha, 0)

    def calculate_final_score(self):
        """计算最终分数"""
        if not self.press_qualities:
            self.score = 0
            self.quality_grade = "No Score"
            return

        avg_score = sum(self.press_qualities) / len(self.press_qualities)
        self.score = int(avg_score)

        # 根据平均分确定等级
        if self.score >= 90:
            self.quality_grade = "Perfect"
        elif self.score >= 75:
            self.quality_grade = "Excellent"
        elif self.score >= 60:
            self.quality_grade = "Good"
        elif self.score >= 40:
            self.quality_grade = "OK"
        else:
            self.quality_grade = "Needs Improvement"

    def get_score(self):
        """返回当前分数"""
        return self.score

    def get_quality_grade(self):
        """返回质量等级"""
        return self.quality_grade

    def is_finished(self):
        return self.state == PizzaState.FINISHED


# ==========================================
#      第二关：抹酱 (Sauce Simulator)
# ==========================================
class TomatoSauceSimulator:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.sauce_points = []
        self.bottle_img = load_sprite('ketchup_bottle.png', target_long=200)  # 调大瓶子尺寸
        self.crust_img = load_sprite('crust.jpg', target_size=(w, h))
        self.bottle_pos = (0, 0)
        self.is_squeezing = False
        self.target_coverage = 100.0
        self.current_coverage = 0.0

        # 评分系统
        self.score = 0
        self.quality_grade = ""
        self.last_quality_msg = ""
        self.last_quality_color = (100, 100, 100)
        self.animation_timer = 0

    def process(self, frame, hand_landmarks):
        display = frame.copy()
        if self.crust_img is not None:
            gray = cv2.cvtColor(self.crust_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            display[mask == 255] = self.crust_img[mask == 255]

        for sx, sy, r in self.sauce_points:
            cv2.circle(display, (sx, sy), r, (0, 0, 200), -1)

        self.current_coverage = len(self.sauce_points) / 2.0

        draw_styled_text(display, f"Phase 2: Sauce ({self.current_coverage:.0f}% / {self.target_coverage:.0f}%)",
                         (30, 50), 1, (0, 255, 255), 2, (0, 0, 0))

        draw_progress_bar(display, 30, 70, 300, 20, self.current_coverage / self.target_coverage, (0, 0, 200),
                          (50, 50, 50), f"{int(self.current_coverage)}%")

        instructions = [
            "Instructions:",
            "1. Use thumb and index finger",
            "2. Pinch to squeeze sauce",
            "3. Cover pizza evenly",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(display, line, (self.w - 450, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not hand_landmarks: return display

        lm = hand_landmarks[0].landmark
        thumb = lm[4]
        index = lm[8]
        dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
        cx = int((thumb.x + index.x) / 2 * self.w)
        cy = int((thumb.y + index.y) / 2 * self.h)
        self.bottle_pos = (cx, cy)

        if dist < 0.05:
            self.is_squeezing = True
            offset_x = np.random.randint(-10, 10)
            offset_y = np.random.randint(60, 90)
            self.sauce_points.append((cx + offset_x, cy + offset_y, 15))
        else:
            self.is_squeezing = False

        if self.bottle_img is not None:
            paste_sprite(display, self.bottle_img, (cx, cy - 50), center=True)
        else:
            cv2.rectangle(display, (cx - 20, cy - 60), (cx + 20, cy), (0, 0, 255), -1)

        # 显示质量信息
        if self.last_quality_msg:
            display = draw_text_with_emoji(display, f"{self.last_quality_msg}", (50, self.h - 90),
                                           font_scale=1.0, color=self.last_quality_color, thickness=3)  # 增大字体和加粗

        return display

    def calculate_final_score(self):
        """计算最终分数"""
        if self.current_coverage < self.target_coverage:
            self.score = 0
            self.quality_grade = "Incomplete"
            return

        # 根据覆盖率计算分数
        coverage_ratio = self.current_coverage / self.target_coverage

        # 考虑覆盖的均匀性（通过检查点的分布）
        if len(self.sauce_points) >= 80:  # 提高perfect的门槛
            self.score = 100
            self.quality_grade = "Perfect"
        elif len(self.sauce_points) >= 60:  # 提高excellent的门槛
            self.score = 90
            self.quality_grade = "Excellent"
        elif len(self.sauce_points) >= 40:  # 提高good的门槛
            self.score = 75
            self.quality_grade = "Good"
        elif len(self.sauce_points) >= 25:
            self.score = 60
            self.quality_grade = "OK"
        else:
            self.score = 40
            self.quality_grade = "Needs Improvement"

    def get_score(self):
        """返回当前分数"""
        return self.score

    def get_quality_grade(self):
        """返回质量等级"""
        return self.quality_grade

    def is_finished(self):
        finished = self.current_coverage >= self.target_coverage
        if finished and self.score == 0:
            self.calculate_final_score()
        return finished


# ==========================================
#      第三关：切菜 (Cutting Simulator) - 优化版
# ==========================================
class PizzaCuttingSimulator:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.total_cuts = 0
        self.target_cuts = 5
        self.knife_xy = None
        self.last_cut = False
        self.finished = False
        self.cut_animation_timer = 0
        self.animation_timer = 0
        self.show_fragment_animation = False
        self.fragment_animation_timer = 0

        self.knife_img = load_sprite('knife.png', target_long=250)
        if self.knife_img is None:
            self.knife_h, self.knife_w = 200, 50
        else:
            self.knife_h, self.knife_w = self.knife_img.shape[:2]

        # 评分系统
        self.score = 0
        self.quality_grade = ""
        self.cut_quality = []  # 记录每次切菜的质量
        self.last_quality_msg = ""
        self.last_quality_color = (100, 100, 100)

        food_names = ['ananas', 'cheese', 'ham', 'mushroom', 'pepper', 'tomato']
        self.food_raw = {}
        self.food_cut = {}
        self.food_coords = {}
        self.cut_coords = {}

        for name in food_names:
            raw = load_sprite(f'{name}.png', target_long=100)
            cut = load_sprite(f'{name}_cut.png', target_long=150)
            if raw is None:
                raw = np.zeros((100, 100, 3), np.uint8)
                raw[:] = (0, 255, 255)
            if cut is None:
                cut = np.zeros((100, 100, 3), np.uint8)
                cut[:] = (0, 100, 100)
            self.food_raw[name] = raw
            self.food_cut[name] = cut

        self.cols, self.rows, self.gap = 3, 2, 80
        self.init_food_positions()

    def init_food_positions(self):
        foods = [('ananas', self.food_raw['ananas'].shape[1], self.food_raw['ananas'].shape[0]),
                 ('cheese', self.food_raw['cheese'].shape[1], self.food_raw['cheese'].shape[0]),
                 ('ham', self.food_raw['ham'].shape[1], self.food_raw['ham'].shape[0]),
                 ('mushroom', self.food_raw['mushroom'].shape[1], self.food_raw['mushroom'].shape[0]),
                 ('pepper', self.food_raw['pepper'].shape[1], self.food_raw['pepper'].shape[0]),
                 ('tomato', self.food_raw['tomato'].shape[1], self.food_raw['tomato'].shape[0])]

        col_w = [max(fw for _, fw, fh in foods[c::self.cols]) for c in range(self.cols)]
        row_h = [max(fh for _, fw, fh in foods[r * self.cols:(r + 1) * self.cols]) for r in range(self.rows)]

        total_w = sum(col_w)
        total_h = sum(row_h)

        start_x = (self.w - total_w) // 2
        start_y = (self.h - total_h) // 2

        y = start_y
        coords = []
        for r in range(self.rows):
            x = start_x
            for c in range(self.cols):
                idx = r * self.cols + c
                coords.append((x, y))
                x += col_w[c]
            y += row_h[r]

        for i, name in enumerate(['ananas', 'cheese', 'ham', 'mushroom', 'pepper', 'tomato']):
            self.food_coords[name] = coords[i]

        gap_cut = 30
        cut_cell_w = max(self.food_cut[name].shape[1] for name in self.food_cut) + gap_cut
        cut_cell_h = max(self.food_cut[name].shape[0] for name in self.food_cut) + gap_cut
        cut_total_w = cut_cell_w * self.cols
        cut_total_h = cut_cell_h * self.rows
        cut_start_x = (self.w - cut_total_w) // 2
        cut_start_y = (self.h - cut_total_h) // 2
        cut_coords = [(cut_start_x + c * cut_cell_w, cut_start_y + r * cut_cell_h)
                      for r in range(self.rows) for c in range(self.cols)]

        for i, name in enumerate(['ananas', 'cheese', 'ham', 'mushroom', 'pepper', 'tomato']):
            self.cut_coords[name] = cut_coords[i]

        self.board_x = coords[0][0]
        self.board_y = coords[0][1]
        self.board_w = sum(col_w)
        self.board_h = sum(row_h)

    def is_fist(self, landmarks):
        """检测握拳手势"""
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Check if fingertips are below PIP joints (finger curled)
        curled_count = 0
        for i in range(5):
            if pts[tips[i]][1] > pts[pips[i]][1]:  # y increases downward
                curled_count += 1

        # Check average distance from palm base (wrist)
        wrist = pts[0]
        dist_avg = sum(np.linalg.norm(pts[tips[i]] - wrist) for i in range(5)) / 5

        # Fist detected: fingers curled AND close to palm
        return curled_count >= 4 and dist_avg < 0.2

    def process(self, frame, hand_landmarks):
        display = frame.copy()

        all_done = self.total_cuts >= self.target_cuts

        for name in ['ananas', 'cheese', 'ham', 'mushroom', 'pepper', 'tomato']:
            if all_done:
                img = self.food_cut[name]
                xy = self.cut_coords[name]
                raw_wh = (self.food_raw[name].shape[1], self.food_raw[name].shape[0])
                off_x = (raw_wh[0] - img.shape[1]) // 2
                off_y = (raw_wh[1] - img.shape[0]) // 2
                paste_sprite(display, img, (xy[0] + off_x, xy[1] + off_y))
            else:
                paste_sprite(display, self.food_raw[name], self.food_coords[name])

        if not all_done:
            cv2.rectangle(display,
                          (self.board_x, self.board_y),
                          (self.board_x + self.board_w, self.board_y + self.board_h),
                          (0, 255, 0), 2)

        draw_styled_text(display, f"Phase 3: Cut Ingredients ({self.total_cuts}/{self.target_cuts})", (30, 50), 1,
                         (0, 255, 255), 2, (0, 0, 0))

        draw_progress_bar(display, 30, 70, 300, 20, self.total_cuts / self.target_cuts, (0, 255, 0), (50, 50, 50),
                          f"{self.total_cuts}/{self.target_cuts}")

        instructions = [
            "Instructions:",
            "1. Make a FIST to grab knife",
            "2. Drag knife over ingredients",
            "3. Move in and out of green box",
            f"4. Need {self.target_cuts} cuts",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(display, line, (self.w - 450, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.finished:
            if self.cut_animation_timer < 60:
                self.cut_animation_timer += 1
                alpha = min(1.0, self.cut_animation_timer / 30.0)
                overlay = display.copy()
                cv2.putText(overlay, "Chopping Done!", (self.w // 2 - 200, self.h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 3)
                cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            else:
                draw_glowing_text(display, "Chopping Done!", (self.w // 2 - 200, self.h // 2), 2, (0, 255, 0), 3)
            return display

        if not hand_landmarks: return display

        lm = hand_landmarks[0].landmark
        is_fist = self.is_fist(lm)
        wrist = lm[0]
        mcp = lm[5]
        fist_center = (int((wrist.x + mcp.x) / 2 * self.w), int((wrist.y + mcp.y) / 2 * self.h))

        if self.knife_xy is None:
            self.knife_xy = (15, 15)

        grab_range = 80
        handle_cx = self.knife_xy[0] + self.knife_w // 4
        handle_cy = self.knife_xy[1] + 3 * self.knife_h // 4

        color = (0, 255, 0) if is_fist else (0, 0, 255)
        cv2.circle(display, (handle_cx, handle_cy), grab_range, (0, 255, 0), 1)
        cv2.circle(display, fist_center, 10, color, -1)

        dist_hand_knife = math.hypot(fist_center[0] - handle_cx, fist_center[1] - handle_cy)

        if is_fist and dist_hand_knife < grab_range:
            self.knife_xy = (fist_center[0] - self.knife_w // 4, fist_center[1] - 3 * self.knife_h // 4)

        paste_sprite(display, self.knife_img, self.knife_xy)

        if self.total_cuts < self.target_cuts:
            kx = self.knife_xy[0] + int(self.knife_w * 0.55)
            ky = self.knife_xy[1] + int(self.knife_h * 0.25)
            kw = int(self.knife_w * 0.45)
            kh = int(self.knife_h * 0.75)

            cut_now = (kx < self.board_x + self.board_w and kx + kw > self.board_x and
                       ky < self.board_y + self.board_h and ky + kh > self.board_y)

            if cut_now and is_fist and not self.last_cut:
                self.total_cuts += 1
                self.last_cut = True

                # 评估切菜质量（根据刀在切割区域内的位置）
                center_x = self.board_x + self.board_w // 2
                center_y = self.board_y + self.board_h // 2
                knife_center_x = kx + kw // 2
                knife_center_y = ky + kh // 2
                dist_from_center = math.hypot(knife_center_x - center_x, knife_center_y - center_y)
                max_dist = math.hypot(self.board_w // 2, self.board_h // 2)

                # 距离中心越近，质量越高
                quality_ratio = max(0, 1 - (dist_from_center / max_dist))
                # 调整评分算法，确保普遍得分在80以上
                quality_score = int(quality_ratio * 40 + 80)  # 最低80分，最高100分
                self.cut_quality.append(quality_score)

                # 显示质量反馈
                if quality_score >= 80:
                    self.last_quality_msg = "Perfect"
                    self.last_quality_color = (0, 255, 0)
                elif quality_score >= 60:
                    self.last_quality_msg = "Good"
                    self.last_quality_color = (0, 255, 255)
                else:
                    self.last_quality_msg = "OK"
                    self.last_quality_color = (0, 165, 255)

            elif not cut_now:
                self.last_cut = False

        # 显示质量信息
        if self.last_quality_msg and self.total_cuts < self.target_cuts:
            display = draw_text_with_emoji(display, f"{self.last_quality_msg}", (50, self.h - 90),
                                           font_scale=1.0, color=self.last_quality_color, thickness=3)  # 增大字体和加粗

        if self.total_cuts >= self.target_cuts:
            if not self.show_fragment_animation:
                self.show_fragment_animation = True
                self.fragment_animation_timer = 0
                # Pre-generate fragments
                self.generated_fragments = {}
                for name in ['ananas', 'cheese', 'ham', 'mushroom', 'pepper', 'tomato']:
                    xy = self.food_coords[name]
                    fragments = []
                    num_fragments = 3
                    for _ in range(num_fragments):
                        angle = np.random.uniform(0, 2 * np.pi)
                        radius = np.random.uniform(20, 60)
                        fragment_pos = (xy[0] + int(radius * np.cos(angle)),
                                        xy[1] + int(radius * np.sin(angle)))
                        fragments.append(fragment_pos)
                    self.generated_fragments[name] = fragments
            elif self.fragment_animation_timer < 60:  # 减少动画时间到60帧（约2秒）
                self.fragment_animation_timer += 1
                for name in ['ananas', 'cheese', 'ham', 'mushroom', 'pepper', 'tomato']:
                    xy = self.food_coords[name]
                    alpha = min(1.0, self.fragment_animation_timer / 30.0)
                    # Show cut fragments in original positions
                    cut_img = self.food_cut[name]
                    paste_sprite(display, cut_img, xy)
                    # Add fragment effect using pre-generated fragments
                    if self.fragment_animation_timer > 15 and name in self.generated_fragments:  # 减少延迟
                        small_img = cv2.resize(cut_img, (0, 0), fx=0.3, fy=0.3)
                        for fragment_pos in self.generated_fragments[name]:
                            paste_sprite(display, small_img, fragment_pos, center=True)
            else:
                self.calculate_final_score()  # 计算最终分数
                self.finished = True

        return display

    def calculate_final_score(self):
        """计算最终分数"""
        if not self.cut_quality:
            self.score = 0
            self.quality_grade = "No Score"
            return

        avg_score = sum(self.cut_quality) / len(self.cut_quality)
        self.score = int(avg_score)

        # 根据平均分确定等级
        if self.score >= 90:
            self.quality_grade = "Perfect"
        elif self.score >= 75:
            self.quality_grade = "Excellent"
        elif self.score >= 60:
            self.quality_grade = "Good"
        elif self.score >= 40:
            self.quality_grade = "OK"
        else:
            self.quality_grade = "Needs Improvement"

    def get_score(self):
        """返回当前分数"""
        return self.score

    def get_quality_grade(self):
        """返回质量等级"""
        return self.quality_grade

    def is_finished(self):
        if self.finished and self.score == 0:
            self.calculate_final_score()
        return self.finished


# ==========================================
#      第四关：撒料 (Topping Simulator)
# ==========================================
class ToppingSimulator:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.finished = False
        self.animation_timer = 0
        self.animation_alpha = 0.0

        self.crust_img = load_sprite('crust.jpg', target_size=(w, h))

        food_names = ['ananas_cut', 'cheese_cut', 'ham_cut', 'mushroom_cut', 'pepper_cut', 'tomato_cut']
        self.toppings = []

        for name in food_names:
            img = load_sprite(f'{name}.png', target_long=250)
            if img is None:
                img = np.zeros((200, 200, 3), np.uint8)
                img[:] = (200, 200, 200)
            self.toppings.append({'name': name, 'img': img, 'placed': False, 'fragments': []})

        self.pizza_center = (w // 2, h // 2)
        self.pizza_radius = 200
        self.current_topping_idx = 0
        self.topping_pos = (w // 2, 120)

        self.is_dragging = False
        self.drag_offset = (0, 0)

        self.message = "Drag toppings onto the pizza!"
        self.message_timer = 0
        self.finish_time = 0

        # 评分系统
        self.score = 0
        self.quality_grade = ""
        self.placement_quality = []  # 记录每次放置的质量

    def set_message(self, text, duration=60):
        self.message = text
        self.message_timer = duration

    def is_fist(self, landmarks):
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        curled_count = 0
        for i in range(5):
            if pts[tips[i]][1] > pts[pips[i]][1]:
                curled_count += 1

        wrist = pts[0]
        dist_avg = sum(np.linalg.norm(pts[tips[i]] - wrist) for i in range(5)) / 5

        return curled_count >= 4 and dist_avg < 0.2

    def process(self, frame, hand_landmarks):
        display = frame.copy()

        # Entrance animation
        if self.animation_timer < 60:
            self.animation_timer += 1
            self.animation_alpha = min(1.0, self.animation_timer / 30.0)

        if self.crust_img is not None:
            gray = cv2.cvtColor(self.crust_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            display[mask == 255] = self.crust_img[mask == 255]

        placed_count = sum(1 for t in self.toppings if t['placed'])
        total_count = len(self.toppings)

        draw_styled_text(display, f"Phase 4: Add Toppings ({placed_count}/{total_count})", (30, 50), 1, (0, 255, 255),
                         2, (0, 0, 0))

        draw_progress_bar(display, 30, 70, 300, 20, placed_count / total_count, (255, 165, 0), (50, 50, 50),
                          f"{placed_count}/{total_count}")

        instructions = [
            "Instructions:",
            "1. Drag ingredient with fist",
            "2. Drag it to pizza center",
            "3. Open hand to place it",
            f"4. Place all {total_count} toppings",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(display, line, (self.w - 450, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.message:
            # 去掉表情符号
            message_text = self.message.split(' ')[0] if ' ' in self.message and any(
                char in '💯👍' for char in self.message) else self.message
            text_size = cv2.getTextSize(message_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (self.w - text_size[0]) // 2
            text_y = self.h - 80  # 移动到屏幕下方
            overlay = display.copy()
            cv2.rectangle(overlay, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            cv2.putText(display, message_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if self.message_timer > 0:
                self.message_timer -= 1

        if self.finished:
            draw_glowing_text(display, "Toppings Complete!", (self.w // 2 - 200, self.h // 2), 2, (0, 255, 0), 3)
            return display

        if not hand_landmarks: return display

        lm = hand_landmarks[0].landmark
        is_fist = self.is_fist(lm)
        wrist = lm[0]
        mcp = lm[5]
        fist_center = (int((wrist.x + mcp.x) / 2 * self.w), int((wrist.y + mcp.y) / 2 * self.h))

        # Draw pizza outline with glow effect
        for i in range(3):
            alpha = 50 - i * 15
            overlay = display.copy()
            cv2.circle(overlay, self.pizza_center, self.pizza_radius + i * 2, (200, 200, 150), 3)
            cv2.addWeighted(overlay, alpha / 100.0, display, 1 - alpha / 100.0, 0, display)
        cv2.circle(display, self.pizza_center, self.pizza_radius, (200, 200, 150), 3)

        # Draw placed toppings as scattered fragments
        for topping in self.toppings:
            if topping['placed']:
                for fragment_pos in topping['fragments']:
                    # Scale down fragments for more natural look (larger than before)
                    scaled_img = cv2.resize(topping['img'], (0, 0), fx=0.6, fy=0.6)
                    paste_sprite(display, scaled_img, fragment_pos, center=True)

        # Draw current unplaced topping
        if self.current_topping_idx < len(self.toppings):
            current_topping = self.toppings[self.current_topping_idx]
            if not current_topping['placed']:
                if self.is_dragging:
                    paste_sprite(display, current_topping['img'],
                                 (fist_center[0] + self.drag_offset[0], fist_center[1] + self.drag_offset[1]),
                                 center=True)
                else:
                    paste_sprite(display, current_topping['img'], self.topping_pos, center=True)

                # Check if hand is near topping
                dist_topping = math.hypot(fist_center[0] - self.topping_pos[0],
                                          fist_center[1] - self.topping_pos[1])

                if is_fist and dist_topping < 80 and not self.is_dragging:
                    self.is_dragging = True
                    self.drag_offset = (self.topping_pos[0] - fist_center[0],
                                        self.topping_pos[1] - fist_center[1])

                elif not is_fist and self.is_dragging:
                    # Check if dropped on pizza
                    dist_pizza = math.hypot(fist_center[0] - self.pizza_center[0],
                                            fist_center[1] - self.pizza_center[1])
                    if dist_pizza < self.pizza_radius:
                        current_topping['placed'] = True

                        # 评估放置质量（距离中心越近，质量越高）
                        placement_quality_ratio = max(0, 1 - (dist_pizza / self.pizza_radius))
                        placement_score = int(placement_quality_ratio * 100)
                        self.placement_quality.append(placement_score)

                        # Generate scattered fragments
                        num_fragments = np.random.randint(5, 10)
                        for _ in range(num_fragments):
                            # Random position on entire pizza
                            angle = np.random.uniform(0, 2 * np.pi)
                            radius = np.random.uniform(0, self.pizza_radius * 0.9)
                            fragment_pos = (self.pizza_center[0] + int(radius * np.cos(angle)),
                                            self.pizza_center[1] + int(radius * np.sin(angle)))
                            current_topping['fragments'].append(fragment_pos)
                        self.current_topping_idx += 1

                        # 显示质量反馈
                        if placement_score >= 80:
                            self.set_message("Perfect placement", 30)
                        elif placement_score >= 60:
                            self.set_message("Good placement", 30)
                        else:
                            self.set_message("OK placement", 30)
                    else:
                        self.set_message("Place it on the pizza!", 30)
                    self.is_dragging = False

        # Check if all toppings placed
        if all(t['placed'] for t in self.toppings):
            if not self.finished:
                self.calculate_final_score()
                self.finished = True
                self.finish_time = time.time()

        return display

    def calculate_final_score(self):
        """计算最终分数"""
        if not self.placement_quality:
            self.score = 0
            self.quality_grade = "No Score"
            return

        avg_score = sum(self.placement_quality) / len(self.placement_quality)
        self.score = int(avg_score)

        # 根据平均分确定等级
        if self.score >= 90:
            self.quality_grade = "Perfect"
        elif self.score >= 75:
            self.quality_grade = "Excellent"
        elif self.score >= 60:
            self.quality_grade = "Good"
        elif self.score >= 40:
            self.quality_grade = "OK"
        else:
            self.quality_grade = "Needs Improvement"

    def get_score(self):
        """返回当前分数"""
        return self.score

    def get_quality_grade(self):
        """返回质量等级"""
        return self.quality_grade

    def is_finished(self):
        if not self.finished:
            return False
        return time.time() - self.finish_time >= 4.0  # 增加停顿时间到4秒，确保所有操作完成


# ==========================================
#      第五关：烘焙 (Baking Simulator)
# ==========================================
class PizzaBakingSimulator:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.finished = False
        self.state = "IDLE"
        self.animation_timer = 0

        self.raw_pizza = load_sprite('pizza_raw.png', target_long=350)
        self.cooked_pizza = load_sprite('pizza_cooked.jpg', target_long=350)
        self.oven = load_sprite('oven.png', target_long=700)

        self.pizza_xy = [w // 4, h // 2 + 50]
        self.oven_xy = [w * 3 // 4, h // 2]

        self.is_dragging = False

        self.bar_x = w // 2 - 200
        self.bar_y = h - 150
        self.bar_width = 400
        self.bar_height = 40
        self.bar_position = 0.0
        self.bar_direction = 1
        self.bar_speed = 0.008

        self.perfect_start = 0.35
        self.perfect_end = 0.65
        self.perfect_width = self.perfect_end - self.perfect_start

        self.score = 0
        self.last_quality_msg = ""
        self.last_quality_color = (100, 100, 100)
        self.finish_time = 0
        self.quality_grade = ""

    def get_score(self):
        """返回当前分数"""
        return self.score

    def get_quality_grade(self):
        """返回质量等级"""
        return self.quality_grade

    def predict_score(self, position):
        if self.perfect_start <= position <= self.perfect_end:
            return 100
        elif position < self.perfect_start:
            return int(100 * (position / self.perfect_start))
        else:
            return int(100 * ((1.0 - position) / (1.0 - self.perfect_end)))

    def is_fist(self, landmarks):
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Check if fingertips are below PIP joints (finger curled)
        curled_count = 0
        for i in range(5):
            if pts[tips[i]][1] > pts[pips[i]][1]:  # y increases downward
                curled_count += 1

        # Check average distance from palm base (wrist)
        wrist = pts[0]
        dist_avg = sum(np.linalg.norm(pts[tips[i]] - wrist) for i in range(5)) / 5

        # Fist detected: fingers curled AND close to palm
        return curled_count >= 4 and dist_avg < 0.2

    def process(self, frame, hand_landmarks):
        display = frame.copy()

        # Entrance animation
        if self.animation_timer < 60:
            self.animation_timer += 1

        if self.oven is not None:
            paste_sprite(display, self.oven, self.oven_xy, center=True)

        draw_styled_text(display, f"Phase 5: Baking Pizza", (30, 50), 1, (0, 255, 255), 2, (0, 0, 0))

        instructions = [
            "Instructions:",
            "1. Drag pizza to oven",
            "2. Wait for green zone",
            "3. Make a FIST to finish",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(display, line, (self.w - 450, 80 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw progress bar with gradient background
        draw_gradient_rect(display, (self.bar_x, self.bar_y),
                           (self.bar_x + self.bar_width, self.bar_y + self.bar_height),
                           (30, 30, 30), (70, 70, 70))

        perfect_x = self.bar_x + int(self.perfect_start * self.bar_width)
        perfect_w = int(self.perfect_width * self.bar_width)

        # Draw perfect zone with gradient
        draw_gradient_rect(display, (perfect_x, self.bar_y),
                           (perfect_x + perfect_w, self.bar_y + self.bar_height),
                           (0, 200, 0), (0, 255, 100))

        cv2.putText(display, "Too Early", (self.bar_x + 20, self.bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        cv2.putText(display, "Good", (self.bar_x + int(self.bar_width * 0.25), self.bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(display, "PERFECT", (self.bar_x + int(self.bar_width * 0.45), self.bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, "Good", (self.bar_x + int(self.bar_width * 0.70), self.bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(display, "Too Late", (self.bar_x + self.bar_width - 80, self.bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

        if self.state == "TIMING":
            self.bar_position += self.bar_speed * self.bar_direction
            if self.bar_position >= 1.0:
                self.bar_position = 1.0
                self.bar_direction = -1
            elif self.bar_position <= 0.0:
                self.bar_position = 0.0
                self.bar_direction = 1

        if self.state == "TIMING":
            bar_x_pos = self.bar_x + int(self.bar_position * self.bar_width)
            # Draw glowing indicator
            for i in range(3):
                alpha = 100 - i * 30
                overlay = display.copy()
                indicator_width = 20 - i * 4
                cv2.rectangle(overlay, (bar_x_pos - indicator_width, self.bar_y),
                              (bar_x_pos + indicator_width, self.bar_y + self.bar_height),
                              (255, 50, 50), -1)
                cv2.addWeighted(overlay, alpha / 100.0, display, 1 - alpha / 100.0, 0, display)
            cv2.rectangle(display, (bar_x_pos - 10, self.bar_y),
                          (bar_x_pos + 10, self.bar_y + self.bar_height),
                          (255, 255, 255), 2)

        if self.finished:
            # 显示熟披萨
            if self.cooked_pizza is not None:
                # 添加发光效果
                for i in range(3):
                    alpha = 30 - i * 10
                    overlay = display.copy()
                    paste_sprite(overlay, self.cooked_pizza, (self.w // 2, self.h // 2), center=True)
                    cv2.addWeighted(overlay, alpha / 100.0, display, 1 - alpha / 100.0, 0, display)
                paste_sprite(display, self.cooked_pizza, (self.w // 2, self.h // 2), center=True)

            # 显示分数和质量
            score_color = (0, 255, 0) if self.score >= 90 else (0, 255, 255) if self.score >= 60 else (0, 165, 255)
            draw_glowing_text(display, f"DELICIOUS! Score: {self.score}/100", (self.w // 2 - 250, self.h // 2 + 180),
                              1.5, score_color, 3)

            # 显示质量等级
            display = draw_text_with_emoji(display, f"{self.last_quality_msg}",
                                           (self.w // 2 - 150, self.h // 2 + 240),
                                           font_scale=1.0, color=self.last_quality_color)

            # 显示披萨完成提示
            draw_glowing_text(display, "PIZZA COMPLETE", (self.w // 2 - 200, self.h // 2 - 200), 2, (0, 255, 0), 3)
        elif self.state == "TIMING":
            info = "WAIT for bar to reach GREEN zone, then CLOSE FIST!"
            cv2.putText(display, info, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            draw_glowing_text(display, "CLOSE FIST NOW!", (self.w // 2 - 150, self.bar_y - 80), 1.2, (255, 255, 0), 3)
        else:
            info = "Grab Pizza (Fist) -> Drop in Oven (Open Hand)"
            cv2.putText(display, info, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if self.state == "TIMING" and not self.finished:
            predicted_score = self.predict_score(self.bar_position)
            score_color = (0, 255, 0) if predicted_score >= 90 else (0, 255, 255) if predicted_score >= 60 else (0, 165,
                                                                                                                 255)
            cv2.putText(display, f"Score: {predicted_score}", (self.bar_x + self.bar_width + 20, self.bar_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)

        if not self.finished:
            if self.state == "IDLE":
                pizza_img = self.raw_pizza
            else:
                alpha = min(1.0, (time.time() - self.start_time) / 3.0)
                if self.cooked_pizza is not None and self.raw_pizza is not None:
                    if self.cooked_pizza.shape == self.raw_pizza.shape:
                        pizza_img = cv2.addWeighted(self.raw_pizza, 1 - alpha, self.cooked_pizza, alpha, 0)
                    else:
                        pizza_img = self.cooked_pizza if alpha > 0.5 else self.raw_pizza
                else:
                    pizza_img = self.raw_pizza if self.raw_pizza is not None else self.cooked_pizza

            if pizza_img is not None:
                paste_sprite(display, pizza_img, self.pizza_xy, center=True)

        if not hand_landmarks: return display

        lm = hand_landmarks[0].landmark
        is_fist = self.is_fist(lm)
        wrist = lm[0]
        mcp = lm[5]
        fist_center = (int((wrist.x + mcp.x) / 2 * self.w), int((wrist.y + mcp.y) / 2 * self.h))

        if self.state == "IDLE":
            dist_pizza = math.hypot(fist_center[0] - self.pizza_xy[0], fist_center[1] - self.pizza_xy[1])
            if is_fist and dist_pizza < 100:
                self.is_dragging = True
            elif not is_fist and self.is_dragging:
                dist_oven = math.hypot(self.pizza_xy[0] - self.oven_xy[0], self.pizza_xy[1] - self.oven_xy[1])
                if dist_oven < 150:
                    self.state = "TIMING"
                    self.start_time = time.time()
                    self.pizza_xy = [self.oven_xy[0], self.oven_xy[1]]
                self.is_dragging = False

            if self.is_dragging:
                self.pizza_xy = [fist_center[0], fist_center[1]]

        elif self.state == "TIMING":
            if is_fist:
                # 检测到握拳，立即计算分数并锁定
                self.score = self.predict_score(self.bar_position)
                if self.score >= 90:
                    self.last_quality_msg = "Perfect"
                    self.last_quality_color = (0, 255, 0)
                    self.quality_grade = "Perfect"
                elif self.score >= 60:
                    self.last_quality_msg = "Good"
                    self.last_quality_color = (0, 255, 255)
                    self.quality_grade = "Good"
                else:
                    self.last_quality_msg = "OK"
                    self.last_quality_color = (0, 165, 255)
                    self.quality_grade = "OK"
                self.finished = True
                self.finish_time = time.time()
                # 立即完成，不需要等待
                return display

        # Visual feedback for fist detection
        if self.state == "TIMING" and not self.finished:
            status_text = "FIST DETECTED!" if is_fist else "OPEN YOUR FIST!"
            status_color = (0, 255, 0) if is_fist else (0, 0, 255)
            cv2.putText(display, status_text, (self.w // 2 - 150, self.bar_y - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        return display

    def is_finished(self):
        if not self.finished:
            return False
        return time.time() - self.finish_time >= 3.0


# ==========================================
#         主程序 (Main Game Loop)
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率为30fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=0  # 使用轻量级模型以提高性能
    )

    current_phase = 0
    phases = [
        PizzaDoughSimulator(w, h),
        TomatoSauceSimulator(w, h),
        PizzaCuttingSimulator(w, h),
        ToppingSimulator(w, h),
        PizzaBakingSimulator(w, h)
    ]

    phase_names = ["Rolling Dough", "Adding Sauce", "Cutting Ingredients", "Adding Toppings", "Baking Pizza"]

    # 评分系统
    phase_scores = [0, 0, 0, 0, 0]
    phase_grades = ["", "", "", "", ""]
    total_score = 0

    transition_alpha = 0.0
    transition_direction = 0
    transition_phase = -1

    # 性能优化：帧率控制
    last_time = time.time()
    target_fps = 30
    frame_time = 1.0 / target_fps

    def draw_score_panel(display, current_phase_idx, scores, grades, total):
        """绘制分数面板"""
        panel_x = w - 260  # 稍微往左移
        panel_y = h - 200  # 稍微往上移
        panel_w = 240
        panel_h = 180

        # 半透明背景
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # 边框
        cv2.rectangle(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)

        # 标题
        cv2.putText(display, "SCORE BOARD", (panel_x + 40, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 各阶段分数
        for i in range(len(phase_names)):
            y_pos = panel_y + 60 + i * 25
            phase_num = i + 1

            if i < current_phase_idx:
                # 已完成的阶段
                color = (0, 255, 0) if scores[i] >= 80 else (0, 255, 255) if scores[i] >= 60 else (255, 165, 0)
                # 去掉表情符号
                grade_text = grades[i].split(' ')[0] if ' ' in grades[i] else grades[i]
                cv2.putText(display, f"P{phase_num}: {scores[i]} - {grade_text}", (panel_x + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif i == current_phase_idx:
                # 当前阶段
                cv2.putText(display, f"P{phase_num}: IN PROGRESS...", (panel_x + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # 未开始的阶段
                cv2.putText(display, f"P{phase_num}: ---", (panel_x + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # 总分
        total_color = (0, 255, 0) if total >= 350 else (0, 255, 255) if total >= 250 else (255, 165, 0)
        cv2.putText(display, f"TOTAL: {total}/500", (panel_x + 10, panel_y + 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, total_color, 2)

    while cap.isOpened():
        # 帧率控制
        current_time = time.time()
        elapsed = current_time - last_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        last_time = time.time()

        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hand_landmarks = results.multi_hand_landmarks

        if hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        if transition_direction != 0:
            if transition_direction == 1:
                transition_alpha = min(1.0, transition_alpha + 0.05)
                if transition_alpha >= 1.0:
                    transition_direction = -1
                    current_phase += 1
            else:
                transition_alpha = max(0.0, transition_alpha - 0.05)
                if transition_alpha <= 0.0:
                    transition_direction = 0
                    transition_phase = -1

            overlay = frame.copy()
            overlay[:] = (0, 0, 0)
            cv2.addWeighted(overlay, transition_alpha, frame, 1 - transition_alpha, 0, frame)

            if transition_phase >= 0 and transition_phase < len(phase_names):
                draw_glowing_text(frame, f"Next: {phase_names[transition_phase]}", (w // 2 - 150, h // 2), 1.5,
                                  (0, 255, 255), 3)
        elif current_phase < len(phases):
            phase = phases[current_phase]
            frame = phase.process(frame, hand_landmarks)

            # 绘制分数面板
            draw_score_panel(frame, current_phase, phase_scores, phase_grades, total_score)

            if phase.is_finished():
                # 保存当前阶段的分数
                phase_scores[current_phase] = phase.get_score()
                phase_grades[current_phase] = phase.get_quality_grade()
                total_score = sum(phase_scores)

                transition_direction = 1
                transition_phase = current_phase
        else:
            # 游戏完成，显示最终结果
            draw_glowing_text(frame, "CONGRATULATIONS! PIZZA COMPLETE!", (w // 2 - 350, h // 2 - 80), 1.5, (0, 255, 0),
                              3)

            # 显示总分和等级
            final_grade = ""
            final_color = (0, 255, 0)
            if total_score >= 450:
                final_grade = "PERFECT"
            elif total_score >= 400:
                final_grade = "Excellent"
            elif total_score >= 350:
                final_grade = "Great"
            elif total_score >= 250:
                final_grade = "Good"
            else:
                final_grade = "Keep Practicing!"
                final_color = (255, 165, 0)

            frame = draw_text_with_emoji(frame, f"Final Score: {total_score}/500 - {final_grade}",
                                         (w // 2 - 200, h // 2 + 20), font_scale=1.2, color=final_color)

            # 调整 "Press ESC to exit" 文字的位置，增加行间距
            cv2.putText(frame, "Press ESC to exit", (w // 2 - 150, h // 2 + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow('Pizza Making Game', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
