import copy, queue, threading, time
import cv2
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(3, 1000)
        self.cap.set(4, 1000)
        self.queue = queue.Queue()
        self.is_running = True
        t = threading.Thread(target=self.read_nonstop)
        t.daemon = True
        t.start()

    def read_nonstop(self):
        while self.is_running:
            ret, img = self.cap.read()
            img = rescale_frame(img, percent=20)
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

start = time.time()
while True:
    oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas, body_angles = util.draw_bodypose(canvas, candidate, subset)

    # TODO: Call PepperController to move joints
    # FIXME: robot.move_joint_by_angle(["LShoulderRoll", "LElbowRoll", "RShoulderRoll", "RElbowRoll"], body_angles, 0.4)
    print("Moving joints [LShoulderRoll, LElbowRoll, RShoulderRoll, RElbowRoll]: ", body_angles)

    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    canvas, is_left_hand_open, is_right_hand_open = util.draw_handpose(canvas, all_hand_peaks, False)

    # TODO: call PepperContoller to open/close hand
    # FIXME: robot.hand("left", !is_left_hand_open)
    # FIXME: robot.hand("right", !is_right_hand_open)
    print("Setting left hand to be ",  "open" if is_left_hand_open else "closed")
    print("Setting right hand to be ",  "open" if is_right_hand_open else "closed")

    cv2.putText(canvas, "Avg FPS: " + str(round((frame_number + 1) / (time.time() - start), 2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Pepper Pose Mirroring by Tomas Trejdl and Vojtech Tilhon | FEL CVUT 2022', canvas)

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
