import copy, queue, threading, time
import cv2
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

from pepper.robot import Pepper

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Initialize Pepper
ip_address = "10.37.1.216"
port = 9559

robot = Pepper(ip_address, port)
robot.autonomous_life_off()
robot.stand()

# Subscribe to camera, 640x480, 30 FPS
robot.subscribe_camera("camera_top", 2, 30)

frame_number = 1

start = time.time()
while True:
    oriImg = robot.get_camera_frame(show=False)
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas, body_angles = util.draw_bodypose(canvas, candidate, subset)

    # Call PepperController to move joints
    robot.move_joint_by_angle(["LShoulderRoll", "LElbowRoll", "RShoulderRoll", "RElbowRoll"], body_angles, 0.4)
    print("Moving joints [LShoulderRoll, LElbowRoll, RShoulderRoll, RElbowRoll]: ", body_angles)

    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    canvas, is_left_hand_open, is_right_hand_open = util.draw_handpose(canvas, all_hand_peaks, False)

    # Call PepperContoller to open/close hand
    robot.hand("left", !is_left_hand_open)
    robot.hand("right", !is_right_hand_open)
    print("Setting left hand to be ",  "open" if is_left_hand_open else "closed")
    print("Setting right hand to be ",  "open" if is_right_hand_open else "closed")

    cv2.putText(canvas, "Avg FPS: " + str(round(frame_number / (time.time() - start), 2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Pepper Pose Mirroring by Tomas Trejdl and Vojtech Tilhon | FEL CVUT 2022', canvas)

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

robot.unsubscribe_camera()
robot.autonomous_life_on()

cv2.destroyAllWindows()
