import cv2
import numpy as np
import math
import time

class TomatoSauceSimulator:
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 椭圆披萨参数
        self.pizza_center = (screen_width // 2, screen_height // 2)
        self.ellipse_a = 280  # 长轴
        self.ellipse_b = 220  # 短轴
        
        # 酱料点
        self.sauce_points = []
        
        # 瓶子
        self.bottle_img = self.load_bottle_image()
        self.bottle_position = None
        self.last_bottle_position = None
        self.bottle_visible = False
        
        # 参数
        self.pinch_threshold = 0.05
        self.line_thickness = 12
    
    def load_bottle_image(self):
        """加载瓶子图片"""
        bottle_img = cv2.imread('ketchup_bottle.png', cv2.IMREAD_UNCHANGED)
        if bottle_img is None:
            # 创建简单瓶子
            h, w = 60, 30
            img = np.zeros((h, w, 4), dtype=np.uint8)
            cv2.rectangle(img, (5, 15), (w-5, h-5), (0, 0, 255, 255), -1)
            cv2.rectangle(img, (8, 8), (w-8, 15), (255, 100, 0, 255), -1)
            cv2.rectangle(img, (w//2-3, 0), (w//2+3, 8), (0, 255, 255, 255), -1)
            return img
        else:
            # 缩小图片
            scale = 0.3
            new_h = int(bottle_img.shape[0] * scale)
            new_w = int(bottle_img.shape[1] * scale)
            return cv2.resize(bottle_img, (new_w, new_h))
    
    def check_pinch(self, landmarks):
        """检测捏合手势"""
        if landmarks is None:
            return False, None
        
        thumb = landmarks[4]
        index = landmarks[8]
        
        distance = math.sqrt(
            (thumb.x - index.x)**2 + 
            (thumb.y - index.y)**2 + 
            (thumb.z - index.z)**2
        )
        
        pinch_point = (
            int((thumb.x + index.x) / 2 * self.screen_width),
            int((thumb.y + index.y) / 2 * self.screen_height)
        )
        
        return distance < self.pinch_threshold, pinch_point
    
    def is_in_ellipse(self, point):
        """检查点是否在椭圆内"""
        x, y = point
        cx, cy = self.pizza_center
        return ((x - cx) ** 2) / (self.ellipse_a ** 2) + ((y - cy) ** 2) / (self.ellipse_b ** 2) <= 1.0
    
    def draw_bottle(self, frame):
        """绘制瓶子"""
        if not self.bottle_visible or self.bottle_position is None:
            return frame
        
        h, w = self.bottle_img.shape[:2]
        x = self.bottle_position[0] - w // 2
        y = self.bottle_position[1] - h - 20
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.screen_width, x1 + w)
        y2 = min(self.screen_height, y1 + h)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        bottle_resized = cv2.resize(self.bottle_img, (x2-x1, y2-y1))
        
        if bottle_resized.shape[2] == 4:
            alpha = bottle_resized[:, :, 3] / 255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    frame[y1:y2, x1:x2, c] * (1 - alpha) + 
                    bottle_resized[:, :, c] * alpha
                )
        
        return frame
    
    def add_sauce_points(self, start, end):
        """添加酱料点"""
        if start is None or end is None:
            return
        
        distance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        if distance < 5:
            return
        
        num_points = max(2, int(distance * 0.4))
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = int(start[0] + (end[0] - start[0]) * t)
            y = int(start[1] + (end[1] - start[1]) * t)
            
            if self.is_in_ellipse((x, y)):
                self.sauce_points.append((x, y, self.line_thickness, time.time()))
    
    def draw_sauce(self, frame):
        """绘制酱料"""
        for point in self.sauce_points:
            x, y, radius, _ = point
            if self.is_in_ellipse((x, y)):
                cv2.circle(frame, (x, y), radius, (0, 0, 220), -1)
        return frame
    
    def calculate_coverage(self):
        """计算覆盖率"""
        if not self.sauce_points:
            return 0.0
        
        grid_size = 12
        total = 0
        covered = 0
        
        a = self.ellipse_a
        b = self.ellipse_b
        
        for x in range(-a, a, grid_size):
            for y in range(-b, b, grid_size):
                if (x ** 2) / (a ** 2) + (y ** 2) / (b ** 2) <= 1.0:
                    total += 1
                    
                    grid_point = (self.pizza_center[0] + x, self.pizza_center[1] + y)
                    
                    for sx, sy, radius, _ in self.sauce_points:
                        if math.sqrt((sx - grid_point[0])**2 + (sy - grid_point[1])**2) <= radius * 1.5:
                            covered += 1
                            break
        
        return (covered / total * 100) if total > 0 else 0.0
    
    def process(self, frame, hand_landmarks, crust_img=None):
        """处理帧"""
        # 绘制饼皮
        if crust_img is not None:
            gray = cv2.cvtColor(crust_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            frame[mask == 255] = crust_img[mask == 255]
        
        # 绘制酱料
        frame = self.draw_sauce(frame)
        
        # 处理手势
        if hand_landmarks:
            is_pinching, pinch_point = self.check_pinch(hand_landmarks)
            
            if is_pinching:
                if not self.bottle_visible:
                    self.bottle_visible = True
                    self.bottle_position = pinch_point
                else:
                    self.add_sauce_points(self.last_bottle_position, self.bottle_position)
                    self.last_bottle_position = self.bottle_position
                    self.bottle_position = pinch_point
            else:
                self.bottle_visible = False
                self.bottle_position = None
                self.last_bottle_position = None
        
        # 绘制瓶子
        frame = self.draw_bottle(frame)
        
        return frame