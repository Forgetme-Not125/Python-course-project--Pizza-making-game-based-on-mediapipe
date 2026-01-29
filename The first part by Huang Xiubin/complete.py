
import cv2
import numpy as np
import mediapipe as mp

from dough import PizzaDoughSimulator   
from tomato import TomatoSauceSimulator

class CompletePizzaSimulator:
    def __init__(self):
        self.screen_width = 1280
        self.screen_height = 720
        
        # 初始化两个阶段
        self.dough_sim = PizzaDoughSimulator(self.screen_width, self.screen_height)
        self.sauce_sim = TomatoSauceSimulator(self.screen_width, self.screen_height)
        
        # 状态
        self.phase = "dough"
        self.dough_completed = False
        
        # 手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 进度目标
        self.target = 50.0
    
    def get_centered_crust(self):
        """获取居中饼皮"""
        if self.dough_sim.crust_img is None:
            return None
            
        crust = self.dough_sim.crust_img.copy()
        h, w = crust.shape[:2]
        
        if h == self.screen_height and w == self.screen_width:
            return crust
        
        centered = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        centered[:] = (150, 150, 150)
        
        start_x = (self.screen_width - w) // 2
        start_y = (self.screen_height - h) // 2
        
        if start_x >= 0 and start_y >= 0 and start_x + w <= self.screen_width and start_y + h <= self.screen_height:
            centered[start_y:start_y+h, start_x:start_x+w] = crust
        
        return centered
    
    def draw_progress_bar(self, frame, coverage):
        """绘制进度条"""
        bar_x = 50
        bar_y = 50
        bar_width = 300
        bar_height = 20
        
        # 进度
        progress = min(1.0, coverage / self.target)
        filled = int(bar_width * progress)
        
        # 背景
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # 填充
        if filled > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 200, 0), -1)
        
        # 文字
        text = f"{coverage:.1f}% / {self.target}%"
        cv2.putText(frame, text, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame):
        """处理视频帧"""
        frame = cv2.flip(frame, 1)
        
        # 手部检测
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hand_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0].landmark
        
        # 根据阶段处理
        if self.phase == "dough":
            output = self.dough_sim.process_frame(frame)
            
            if self.dough_sim.state.name == "FINISHED" and not self.dough_completed:
                self.dough_completed = True
                self.phase = "sauce"
        
        else:
            output = frame.copy()
            
            # 获取饼皮
            crust = self.get_centered_crust()
            
            # 处理酱料
            if crust is not None:
                output = self.sauce_sim.process(output, hand_landmarks, crust)
            else:
                output = self.sauce_sim.process(output, hand_landmarks)
            
            # 显示进度
            coverage = self.sauce_sim.calculate_coverage()
            output = self.draw_progress_bar(output, coverage)
            # 涂酱完成提示（最简单的一行代码）
            if coverage >= self.target:
                cv2.putText(output, "DONE!", (self.screen_width//2 - 50, self.screen_height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        return output
    
    def run(self):
        """运行主程序"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)
        
        print("Pizza Simulator")
        print("Phase 1: Roll dough")
        print("Phase 2: Squeeze sauce (goal: 50%)")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output = self.process_frame(frame)
            cv2.imshow('Pizza Simulator', output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    simulator = CompletePizzaSimulator()
    simulator.run()