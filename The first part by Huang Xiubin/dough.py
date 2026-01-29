import cv2
import numpy as np
import mediapipe as mp
import math
import time
from enum import Enum

# 初始化MediaPipe手部检测
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

class PizzaState(Enum):
    WAITING = 1      # 等待手掌张开
    OVER_DOUGH = 2   # 手掌在面团上方
    PRESSING = 3     # 按压擀饼中
    FINISHED = 4     # 擀饼完成

class PizzaDoughSimulator:
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 初始化状态
        self.state = PizzaState.WAITING
        self.dough_thickness = 1.0  # 初始饼皮厚度 (1.0表示正常)
        self.pizza_radius = 200     # 披萨基准半径
        self.pizza_center = (screen_width // 2, screen_height // 2)
        
        # 擀饼相关变量
        self.last_palm_position = None
        self.pressing_start_time = None
        self.press_count = 0
        self.required_presses = 10  # 增加所需按压次数，使过程变慢
        self.press_strength_history = []
        
        # 创建纯色背景（灰色）
        self.background_color = (150, 150, 150)  # BGR格式，灰色
        self.background_img = np.full((screen_height, screen_width, 3), 
                                    self.background_color, dtype=np.uint8)
        
        # 尝试加载自定义图片
        self.use_custom_images = False
        
        # 加载面团图片
        self.dough_img = cv2.imread('dough.jpg')  # 替换为你的面团图片文件名
        if self.dough_img is not None:
            self.dough_img = cv2.resize(self.dough_img, (screen_width, screen_height))
            self.use_custom_images = True
            print("成功加载面团图片")
        else:
            print("无法加载面团图片，请检查文件路径")
            # 创建简单的面团图片作为备用
            self.dough_img = self.create_simple_dough_image()
        
        # 加载饼皮图片
        self.crust_img = cv2.imread('crust.jpg')  # 替换为你的饼皮图片文件名
        if self.crust_img is not None:
            self.crust_img = cv2.resize(self.crust_img, (screen_width, screen_height))
            self.use_custom_images = True
            print("成功加载饼皮图片")
        else:
            print("无法加载饼皮图片，请检查文件路径")
            # 创建简单的饼皮图片作为备用
            self.crust_img = self.create_simple_crust_image()
        
        # 当前显示的面团图像
        self.current_dough = self.dough_img.copy()
        
        # 擀饼方向追踪
        self.stretch_direction = None
        self.stretch_factor = 1.0  # 拉伸因子
        
        # 提示信息
        self.message = "Please open your palm and move it above the dough to start rolling."
        self.message_timer = 10000
        
        # 擀饼进度
        self.progress = 0.0
        
        # 上一次按压时间（用于控制速度）
        self.last_press_time = 0
        self.min_press_interval = 0.8  # 增加最小按压间隔（秒），防止重复计数
        
        # 手掌与面团中心的距离阈值
        self.distance_threshold = 180  # 增加距离阈值
        
        # 按压质量相关变量
        self.last_press_quality = ""
        self.last_press_quality_color = (0, 255, 0)
        self.last_press_timer = 0
        self.press_quality_history = []  # 记录历史评价
        
        # 防止重复按压的标记
        self.press_in_progress = False
        self.press_start_position = None
        
    def create_simple_dough_image(self):
        """创建简单面团图像（备用）"""
        img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        cv2.circle(img, self.pizza_center, self.pizza_radius, (240, 230, 210), -1)
        cv2.circle(img, self.pizza_center, self.pizza_radius, (200, 190, 170), 3)
        return img
    
    def create_simple_crust_image(self):
        """创建简单饼皮图像（备用）"""
        img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        cv2.circle(img, self.pizza_center, int(self.pizza_radius * 1.3), (255, 240, 220), -1)
        cv2.circle(img, self.pizza_center, int(self.pizza_radius * 1.3), (220, 200, 180), 3)
        return img
    
    def calculate_palm_openness(self, landmarks):
        """计算手掌张开程度"""
        # 使用拇指和食指指尖的距离作为手掌张开的指标
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # 计算指尖距离
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        ) * self.screen_width
        
        # 手掌张开阈值
        open_threshold = 70  # 降低阈值，更容易检测
        return distance > open_threshold, distance
    
    def get_palm_center(self, landmarks):
        """获取手掌中心位置"""
        # 使用手掌底部和手指根部的点计算中心
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        palm_x = (wrist.x + middle_mcp.x) / 2
        palm_y = (wrist.y + middle_mcp.y) / 2
        
        return (
            int(palm_x * self.screen_width),
            int(palm_y * self.screen_height)
        )
    
    def calculate_distance_to_center(self, palm_position):
        """计算手掌与面团中心的距离"""
        if palm_position is None:
            return float('inf')
        
        distance = math.sqrt(
            (palm_position[0] - self.pizza_center[0])**2 + 
            (palm_position[1] - self.pizza_center[1])**2
        )
        
        return distance
    
    def calculate_press_strength(self, start_pos, end_pos):
        """计算按压力度（基于手掌移动幅度）"""
        if start_pos is None or end_pos is None:
            return 0
        
        # 主要计算垂直移动幅度
        vertical_distance = abs(end_pos[1] - start_pos[1])
        
        # 根据距离计算力度
        max_distance = 150  # 最大参考距离
        strength = min(vertical_distance / max_distance, 1.0)
        return strength
    
    def stretch_dough(self, direction, strength):
        """根据方向和力度拉伸面团"""
        # 缓慢增加进度
        progress_increment = strength * 0.1  # 减小增量使过程变慢
        
        # 根据力度给出评价
        if strength >= 0.8:
            self.last_press_quality = f"Perfect! 💯(+{progress_increment*100:.1f}%)"
            quality_color = (0, 255, 0)  # 绿色
        elif strength >= 0.6:
            self.last_press_quality = f"Good! 👍(+{progress_increment*100:.1f}%)"
            quality_color = (0, 200, 100)  # 黄绿色
        elif strength >= 0.4:
            self.last_press_quality = f"OK 👌(+{progress_increment*100:.1f}%)"
            quality_color = (0, 165, 255)  # 橙色
        elif strength >= 0.2:
            self.last_press_quality = f"Weak 😕(+{progress_increment*100:.1f}%)"
            quality_color = (0, 100, 255)  # 红色
        else:
            self.last_press_quality = f"Too weak! ❌(+{progress_increment*100:.1f}%)"
            quality_color = (0, 0, 255)  # 深红色
        
        # 保存评价和颜色
        self.last_press_quality_color = quality_color
        self.last_press_timer = 30  # 显示30帧

        # 记录到历史
        self.press_quality_history.append(self.last_press_quality)
        # 只保留最近10次
        if len(self.press_quality_history) > 10:
            self.press_quality_history.pop(0)

        # 更新面团厚度 - 力度越小，厚度减少越少
        thickness_reduction = 0.03 + strength * 0.15
        self.dough_thickness = max(0.2, self.dough_thickness - thickness_reduction)
        
        # 更新进度
        self.progress = min(1.0, self.progress + progress_increment)
        
        # 更新拉伸因子
        if direction == "left":
            self.stretch_factor = min(1.5, self.stretch_factor + 0.02 + strength * 0.05)
        elif direction == "right":
            self.stretch_factor = min(1.5, self.stretch_factor + 0.01 + strength * 0.03)
        
        # 生成新的面团图像
        self.update_dough_image()
    
    def update_dough_image(self):
        """根据当前状态更新面团图像"""
        if self.progress >= 1.0:
            # 使用完成的饼皮图像
            self.current_dough = self.crust_img.copy()
            return
        
        # 使用图片混合效果
        if self.use_custom_images and self.dough_img is not None and self.crust_img is not None:
            # 根据进度混合面团和饼皮图片
            blend_factor = self.progress
            
            # 创建拉伸效果
            if self.stretch_factor > 1.0:
                # 计算缩放
                scale_x = 1.0 + (self.stretch_factor - 1.0) * 0.3
                scale_y = 1.0 - (self.stretch_factor - 1.0) * 0.1
                
                # 对面团进行拉伸变换
                M = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
                stretched_dough = cv2.warpAffine(self.dough_img, M, 
                                                (self.screen_width, self.screen_height))
                
                # 创建饼皮遮罩（只取饼皮的非黑色部分）
                gray_crust = cv2.cvtColor(self.crust_img, cv2.COLOR_BGR2GRAY)
                _, crust_mask = cv2.threshold(gray_crust, 20, 255, cv2.THRESH_BINARY)
                
                # 将遮罩转为0-1的浮点数
                crust_mask_float = crust_mask.astype(np.float32) / 255.0
                
                # 创建混合遮罩（根据进度调整）
                blend_mask = crust_mask_float * blend_factor
                
                # 智能混合：只混合饼皮的非黑色部分
                self.current_dough = stretched_dough.copy().astype(np.float32)
                for c in range(3):  # 对BGR三个通道分别处理
                    self.current_dough[:,:,c] = (
                        self.current_dough[:,:,c] * (1 - blend_mask) + 
                        self.crust_img[:,:,c].astype(np.float32) * blend_mask
                    )
                self.current_dough = self.current_dough.astype(np.uint8)
            else:
                # 非拉伸状态的智能混合
                gray_crust = cv2.cvtColor(self.crust_img, cv2.COLOR_BGR2GRAY)
                _, crust_mask = cv2.threshold(gray_crust, 20, 255, cv2.THRESH_BINARY)
                crust_mask_float = crust_mask.astype(np.float32) / 255.0
                blend_mask = crust_mask_float * blend_factor
                
                self.current_dough = self.dough_img.copy().astype(np.float32)
                for c in range(3):
                    self.current_dough[:,:,c] = (
                        self.current_dough[:,:,c] * (1 - blend_mask) + 
                        self.crust_img[:,:,c].astype(np.float32) * blend_mask
                    )
                self.current_dough = self.current_dough.astype(np.uint8)
        else:
            # 使用生成的图片
            self.current_dough = self.create_simple_dough_image()
    
    def check_thickness_warning(self, strength):
        """检查饼皮厚度并给出提示"""
        if strength < 0.3:  # 力度太小
            self.message = "Rolling force is too small! Please increase the up and down swinging amplitude."
            self.message_timer = 45  # 显示45帧
            return True
        elif self.dough_thickness > 0.7:
            self.message = "The crust is too thick! Please continue rolling."
            self.message_timer = 45
            return True
        return False
    
    def process_frame(self, frame):
        """处理视频帧"""
        # 翻转帧以便镜像显示
        #frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理手部检测
        results = hands.process(frame_rgb)
        
        # 创建显示图像
        display_img = frame.copy()
        
        # 显示面团/饼皮
        dough_display = display_img.copy()
        if self.current_dough is not None:
            gray = cv2.cvtColor(self.current_dough, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            mask_bool = mask == 255
            dough_display[mask_bool] = self.current_dough[mask_bool]
        
        # 绘制面团中心区域（手掌需要到达的区域）
        cv2.circle(dough_display, self.pizza_center, self.distance_threshold, (255, 200, 100), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                mp_drawing.draw_landmarks(
                    dough_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 获取手掌张开状态和中心位置
                palm_open, openness_value = self.calculate_palm_openness(hand_landmarks.landmark)
                palm_center = self.get_palm_center(hand_landmarks.landmark)
                
                # 计算手掌与面团中心的距离
                distance_to_center = self.calculate_distance_to_center(palm_center)
                
                # 在手掌中心绘制圆圈
                circle_color = (0, 255, 0) if palm_open else (0, 0, 255)
                cv2.circle(dough_display, palm_center, 15, circle_color, -1)
                
                # 绘制手掌到面团中心的连线
                cv2.line(dough_display, palm_center, self.pizza_center, (255, 255, 0), 2)
                
                # 显示距离
                cv2.putText(dough_display, f"distance: {int(distance_to_center)}px", 
                           (palm_center[0] + 20, palm_center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 状态机处理
                if self.state == PizzaState.WAITING:
                    if palm_open:
                        if distance_to_center <= self.distance_threshold:
                            self.state = PizzaState.OVER_DOUGH
                            self.message = "Palm is above the dough! Start swinging up and down to roll."
                            self.message_timer = 30
                        else:
                            self.message = "Please move your palm above the dough"
                            self.message_timer = 30
                
                elif self.state == PizzaState.OVER_DOUGH:
                    # 检查手掌是否仍在面团上方
                    if distance_to_center <= self.distance_threshold:
                        if palm_open:
                            # 添加确认时间（例如：保持1秒）
                            current_time = time.time()
                            if not hasattr(self, 'over_dough_start_time'):
                                self.over_dough_start_time = current_time
                            
                            # 检查是否保持了足够时间
                            if current_time - self.over_dough_start_time > 1.0:  # 1秒确认
                                self.state = PizzaState.PRESSING
                                self.last_palm_position = palm_center
                                self.pressing_start_time = time.time()
                                self.message = "Start swinging your palm up and down to press and roll!"
                                self.message_timer = 30
                                # 绘制起始位置
                                cv2.circle(dough_display, palm_center, 10, (255, 0, 0), -1)
                                # 重置计时器
                                delattr(self, 'over_dough_start_time')
                        else:
                            # 手掌闭合，回到等待状态
                            self.state = PizzaState.WAITING
                            if hasattr(self, 'over_dough_start_time'):
                                delattr(self, 'over_dough_start_time')
                    else:
                        # 手掌离开面团区域
                        self.state = PizzaState.WAITING
                        if hasattr(self, 'over_dough_start_time'):
                            delattr(self, 'over_dough_start_time')
                        self.message = "Palm has left the dough."
                        self.message_timer = 30
                
                elif self.state == PizzaState.PRESSING:
                    # 检查手掌是否仍在面团上方
                    if distance_to_center <= self.distance_threshold:
                        if palm_open:
                            current_time = time.time()
                            
                            # 检查是否有足够的垂直移动（上下挥动）
                            if self.last_palm_position:
                                vertical_movement = palm_center[1] - self.last_palm_position[1]
                                vertical_distance = abs(vertical_movement)  # 绝对值用于计算距离
                                
                                # 开始向下移动
                                if vertical_movement > 30 and not self.press_in_progress:
                                    self.press_in_progress = True
                                    self.press_start_position = self.last_palm_position
                                    self.message = "Pressing down..."
                                
                                # 完成向下移动并开始返回（完成一次按压）
                                elif vertical_movement < -20 and self.press_in_progress and self.press_start_position:
                                    self.press_in_progress = False
                                    
                                    # 计算从开始到返回的整个移动
                                    total_distance = abs(palm_center[1] - self.press_start_position[1])
                                    
                                    # 控制按压速度：检查时间间隔
                                    if (total_distance > 30 and palm_open and  # 增加移动距离要求
                                        current_time - self.last_press_time > self.min_press_interval):
                                        
                                        # 计算力度
                                        strength = self.calculate_press_strength(self.press_start_position, palm_center)
                                        self.press_strength_history.append(strength)
                                        
                                        # 计算水平移动方向
                                        horizontal_movement = palm_center[0] - self.press_start_position[0]
                                        direction = "left" if horizontal_movement < 0 else "right"
                                        
                                        # 拉伸面团
                                        self.stretch_dough(direction, strength)
                                        
                                        # 检查厚度警告
                                        self.check_thickness_warning(strength)
                                        
                                        # 更新计数
                                        self.press_count += 1
                                        
                                        # 绘制移动轨迹
                                        cv2.line(dough_display, self.press_start_position, palm_center, (0, 255, 255), 3)
                                        
                                        # 更新时间和位置
                                        self.last_press_time = current_time
                                        self.last_palm_position = palm_center
                                        
                                        # 显示力度值
                                        cv2.putText(dough_display, f"strength: {strength:.2f}", 
                                                   (palm_center[0] + 20, palm_center[1] + 30), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        
                                        # 检查是否完成
                                        if self.press_count >= self.required_presses:
                                            self.state = PizzaState.FINISHED
                                            self.message = "Rolling complete! The perfect crust is ready."
                                            self.message_timer = 60
                                    elif total_distance <= 30:
                                        self.message = "!!!The swing amplitude is not sufficient. Please increase the movement."
                                        self.message_timer = 20
                                
                                # 更新手掌位置用于下一次检测
                                self.last_palm_position = palm_center
                        
                        # 如果手掌闭合，回到等待状态
                        elif not palm_open:
                            self.state = PizzaState.WAITING
                            self.message = "Palm closed. Please open your palm."
                            self.message_timer = 30
                            self.press_in_progress = False  # 重置按压状态
                    else:
                        self.state = PizzaState.WAITING
                        self.message = "The palm has left the dough"
                        self.message_timer = 30
                        self.press_in_progress = False  # 重置按压状态
                    
                    # 显示擀饼进度
                    progress_bar_width = 300
                    progress_filled = int(progress_bar_width * (self.press_count / self.required_presses))
                    cv2.rectangle(dough_display, (50, 50), (50 + progress_bar_width, 70), (100, 100, 100), -1)
                    cv2.rectangle(dough_display, (50, 50), (50 + progress_filled, 70), (0, 200, 0), -1)
                    cv2.putText(dough_display, f"progress of rolling: {self.press_count}/{self.required_presses}", 
                               (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                elif self.state == PizzaState.FINISHED:
                    # 显示最终饼皮
                    self.current_dough = self.crust_img.copy()
                    dough_display = frame.copy()
                    gray = cv2.cvtColor(self.current_dough, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                    mask_bool = mask == 255
                    dough_display[mask_bool] = self.current_dough[mask_bool]
                    # 在手部绘制关键点
                    mp_drawing.draw_landmarks(
                        dough_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 显示状态信息
        state_colors = {
            PizzaState.WAITING: (0, 165, 255),
            PizzaState.OVER_DOUGH: (255, 165, 0),
            PizzaState.PRESSING: (0, 255, 255),
            PizzaState.FINISHED: (0, 255, 0)
        }
        
        cv2.putText(dough_display, f"state: {self.state.name}", 
                   (50, self.screen_height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_colors[self.state], 2)
        
        # 显示最近一次按压评价
        if self.last_press_timer > 0:
            cv2.putText(dough_display, f"Press quality: {self.last_press_quality}", 
                    (50, self.screen_height - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    self.last_press_quality_color, 2)
            self.last_press_timer -= 1
        else:
            # 没有最近按压时显示提示
            cv2.putText(dough_display, "Press quality: Waiting for press...", 
                    (50, self.screen_height - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # 显示进度信息
        cv2.putText(dough_display, f"completeness: {self.progress*100:.1f}%", 
                (50, self.screen_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        
        # 显示提示信息
        if self.message_timer > 0:
            # 添加半透明背景
            text_size = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(dough_display, 
                         (self.screen_width // 2 - text_size[0] // 2 - 10, 20),
                         (self.screen_width // 2 + text_size[0] // 2 + 10, 60),
                         (0, 0, 0, 180), -1)
            
            cv2.putText(dough_display, self.message, 
                       (self.screen_width // 2 - text_size[0] // 2, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        instructions = [
            "Instructions:",
            "1. Open your hand, move to the dough center",
            "2. Wave hand up/down above dough to press",
            f"3. Complete {self.required_presses} effective presses",
        ]
        
        for i, line in enumerate(instructions):
            cv2.putText(dough_display, line, 
                       (self.screen_width - 500, 50 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return dough_display

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 创建披萨擀饼模拟器
    simulator = PizzaDoughSimulator()
    
    print("披萨擀饼模拟系统启动!")
    print("请确保摄像头可以清晰看到您的手掌")
    print("按'q'键退出程序")
    print("注意：需要将面团图片(dough.jpg)和饼皮图片(crust.jpg)放在同一目录下")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 处理帧
        output_frame = simulator.process_frame(frame)
        
        # 显示结果
        cv2.imshow('pizza!', output_frame)
        
        # 检查退出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()