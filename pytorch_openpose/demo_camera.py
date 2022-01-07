import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body
from src.hand import Hand

import queue, threading
import math

edges = {(0, 1), (0, 15), (0, 16), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 12), (9, 10), (10, 11), (11, 22), (11, 24), (12, 13), (13, 14), (14, 19), (14, 21), (15, 17), (16, 18), (19, 20), (22, 23)}

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

# print(f"Torch device: {torch.cuda.get_device_name()}")

class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.queue = queue.Queue()
        self.is_running = True
        t = threading.Thread(target=self.read_nonstop)
        t.daemon = True
        t.start()

    def read_nonstop(self):
        while self.is_running:
            ret, img = self.cap.read()
            if not ret:
                print("ERROR: Reading video capture failed")
                self.is_running = False
                break

            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            self.queue.put(img)

    def read(self):
        return self.queue.get()

    def release(self):
        self.is_running = False
        self.cap.release()

cap = BufferlessVideoCapture(0)

frame_number = 0

def measure_hand_segment_lengths(measure_left):
    hand_segment_lengths = None

    print("Measuring " + ("left" if measure_left else "right") + " hand")
    done_measuring = False
    while not done_measuring or not hand_segment_lengths:
        oriImg = cap.read()
        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # detect hand
        hands_list = util.handDetect(candidate, subset, oriImg)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            if is_left != measure_left:
                continue

            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)

            hand_segment_lengths = util.calculate_hand_segment_lengths(peaks)
            if hand_segment_lengths:
                print("Hand successfully measured")

        canvas = util.draw_handpose(canvas, all_hand_peaks, True)

        cv2.imshow(("Left" if measure_left else "Right") + " hand calibration", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done_measuring = True

    return hand_segment_lengths

while True:
    oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    print("Detecting hand")
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)
    print("Done detecting hand")

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        print("Have hand")
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        print("Done estimating hand")
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    print(all_hand_peaks)
    canvas = util.draw_handpose(canvas, all_hand_peaks, True)

    canvas = cv2.flip(canvas, 1)
    cv2.imshow('demo', canvas)#一个窗口用以显示原视频
    # cv2.imwrite('frame'+str(frame_number)+'.png', canvas)

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
