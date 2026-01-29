# -*- coding: utf-8 -*-
import os
import cv2
import mediapipe as mp
import numpy as np

# ---------- 配置 ----------
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands
MAX_HANDS  = 2
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP  = [2, 5, 9, 13, 17]
FIST_THRESHOLD = 0.15
GRAB_RANGE     = 80
TARGET_CUT     = 5          # 全局 5 刀
COLS, ROWS, GAP = 3, 2, 120

# ---------- 工具 ----------
def load_sprite(img_path, target_long):
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f'缺少素材 {img_path}')
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'无法读取 {img_path}')
    scale = target_long / max(img.shape[:2])
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, img.shape[0], img.shape[1]

def paste_sprite(frame, sprite, xy):
    x, y = xy
    h, w = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + sw), min(h, y + sh)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    spr = sprite[y1-y:y2-y, x1-x:x2-x]
    alpha = spr[:, :, 3:4] / 255.0
    roi[:] = (1 - alpha) * roi + alpha * spr[:, :, :3]

# ---------- 加载素材 ----------
knife_img, knife_h, knife_w = load_sprite('knife.png', 350)

ananas_img,  a_h, a_w = load_sprite('ananas.png',  knife_h//2)
cheese_img,  c_h, c_w = load_sprite('cheese.png',  knife_h//2)
ham_img,     h_h, h_w = load_sprite('ham.png',     knife_h//2)
mushroom_img,m_h, m_w = load_sprite('mushroom.png',knife_h//2)
pepper_img,  p_h, p_w = load_sprite('pepper.png',  knife_h//2)
tomato_img,  t_h, t_w = load_sprite('tomato.png',  knife_h//2)

ananas_cut_img,  ac_h, ac_w = load_sprite('ananas_cut.png',  knife_h)
cheese_cut_img,  cc_h, cc_w = load_sprite('cheese_cut.png',  knife_h)
ham_cut_img,     hc_h, hc_w = load_sprite('ham_cut.png',     knife_h)
mushroom_cut_img,mc_h, mc_w = load_sprite('mushroom_cut.png',knife_h)
pepper_cut_img,  pc_h, pc_w = load_sprite('pepper_cut.png',  knife_h)
tomato_cut_img,  tc_h, tc_w = load_sprite('tomato_cut.png',  knife_h)

# ---------- 初始化 ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('摄像头分辨率:', w, 'x', h)

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=MAX_HANDS,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# 全局状态
knife_xy   = None
total_cuts = 0
last_cut   = False

# ---------- 主循环 ----------
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 手势解析
    is_fist = False
    fist_center = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            dist = sum(np.linalg.norm(pts[FINGER_TIPS[i]] - pts[FINGER_MCP[i]]) for i in range(5)) / 5
            is_fist = dist < FIST_THRESHOLD
            wrist, mcp = pts[0], pts[5]
            fist_center = (int((wrist[0]+mcp[0])/2 * w), int((wrist[1]+mcp[1])/2 * h))
            break

    # 首次摆放食材与刀
    if knife_xy is None:
        knife_xy = (15, 15)          
        foods = [('ananas',a_w,a_h), ('cheese',c_w,c_h), ('ham',h_w,h_h),
                 ('mushroom',m_w,m_h), ('pepper',p_w,p_h), ('tomato',t_w,t_h)]
        # 1. 每列宽度 = 该列最大食材宽，每行高度 = 该行最大食材高
        col_w = [max(fw for _,fw,fh in foods[c::COLS]) for c in range(COLS)]
        row_h = [max(fh for _,fw,fh in foods[r*COLS:(r+1)*COLS]) for r in range(ROWS)]
        # 2. 总宽高
        total_w = sum(col_w)
        total_h = sum(row_h)
        # 3. 起点（整体居中）
        start_x = (w - total_w) // 2
        start_y = (h - total_h) // 2
        # 4. 生成每个食材左上角坐标
        y = start_y
        coords = []
        for r in range(ROWS):
            x = start_x
            for c in range(COLS):
                idx = r * COLS + c
                coords.append((x, y))          # 记录当前食材位置
                x += col_w[c]                  
            y += row_h[r]                      
        ananas_xy, cheese_xy, ham_xy, mushroom_xy, pepper_xy, tomato_xy = coords

    ananas_xy, cheese_xy, ham_xy, mushroom_xy, pepper_xy, tomato_xy = coords

    GAP_CUT = 30          
    cut_cell_w = max(ac_w, cc_w, hc_w, mc_w, pc_w, tc_w) + GAP_CUT
    cut_cell_h = max(ac_h, cc_h, hc_h, mc_h, pc_h, tc_h) + GAP_CUT
    cut_total_w = cut_cell_w * COLS
    cut_total_h = cut_cell_h * ROWS
    cut_start_x = (w - cut_total_w) // 2
    cut_start_y = (h - cut_total_h) // 2
    cut_coords = [(cut_start_x + c * cut_cell_w,
                   cut_start_y + r * cut_cell_h)
                  for r in range(ROWS) for c in range(COLS)]
    ananas_cut_xy, cheese_cut_xy, ham_cut_xy, mushroom_cut_xy, pepper_cut_xy, tomato_cut_xy = cut_coords

    # 刀跟随
    if fist_center is not None:
        handle_cx = knife_xy[0] + knife_w//4
        handle_cy = knife_xy[1] + 3*knife_h//4
        if is_fist and np.linalg.norm([fist_center[0]-handle_cx, fist_center[1]-handle_cy]) < GRAB_RANGE:
            knife_xy = (fist_center[0] - knife_w//4, fist_center[1] - 3*knife_h//4)
    paste_sprite(frame, knife_img, knife_xy)

    # ===== 全局 5 刀交互 =====
    # ===== 1. 定义案板矩形（左上角 + 宽高）=====
    # 用食材整体区域当案板
    board_x, board_y = ananas_xy               # 最左上角
    board_w = sum(col_w)                       # 总列宽
    board_h = sum(row_h)                       # 总行高

    # ===== 2. 交互：刀尖 vs 案板 =====
    all_done = total_cuts >= TARGET_CUT
    if total_cuts < TARGET_CUT:
        # 刀尖矩形
        kx, ky = knife_xy[0] + int(knife_w*0.55), knife_xy[1] + int(knife_h*0.25)
        kw, kh = int(knife_w*0.45), int(knife_h*0.75)
        # 与案板相交判定
        cut_now = (kx < board_x + board_w and kx + kw > board_x and
                   ky < board_y + board_h and ky + kh > board_y)
        if cut_now and is_fist and not last_cut:
            total_cuts += 1
            last_cut = True
        elif not cut_now:
            last_cut = False

    # ===== 统一渲染 =====
    def paste_ing(xy, raw, raw_wh, cut, cut_wh, cut_xy):
        x, y = (cut_xy if all_done and cut is not None else xy)
        img  = cut if all_done and cut is not None else raw
        off_x = (raw_wh[0] - img.shape[1]) // 2
        off_y = (raw_wh[1] - img.shape[0]) // 2
        paste_sprite(frame, img, (x + off_x, y + off_y))

    paste_ing(ananas_xy,   ananas_img,  (a_w, a_h),  ananas_cut_img,  (ac_w, ac_h),  ananas_cut_xy)
    paste_ing(cheese_xy,   cheese_img,  (c_w, c_h),  cheese_cut_img,  (cc_w, cc_h),  cheese_cut_xy)
    paste_ing(ham_xy,      ham_img,     (h_w, h_h),  ham_cut_img,     (hc_w, hc_h),  ham_cut_xy)
    paste_ing(mushroom_xy, mushroom_img,(m_w, m_h),  mushroom_cut_img,(mc_w, mc_h),  mushroom_cut_xy)
    paste_ing(pepper_xy,   pepper_img,  (p_w, p_h),  pepper_cut_img,  (pc_w, pc_h),  pepper_cut_xy)
    paste_ing(tomato_xy,   tomato_img,  (t_w, t_h),  tomato_cut_img,  (tc_w, tc_h),  tomato_cut_xy)

    # 全局提示
    cv2.putText(frame, f'{total_cuts}/{TARGET_CUT}',
                (w//2-40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,0), 3)

    cv2.imshow('Grab knife – global 5 cuts', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------- 清理 ----------
cap.release()
cv2.destroyAllWindows()
hands.close()