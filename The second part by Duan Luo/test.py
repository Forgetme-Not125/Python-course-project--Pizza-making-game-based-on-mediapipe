import cv2, time

# 1. 创建时就指定后端
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # 关键在这里！

# 2. 再降分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 60)

t0 = time.time()
ret, frame = cap.read()
print('read 一次耗时:', (time.time()-t0)*1000, 'ms')
print('分辨率:', frame.shape[1], 'x', frame.shape[0])
cap.release()