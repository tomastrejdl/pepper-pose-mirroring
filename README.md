# Pepper Pose Mirroring

Semester project for KSY (Kognitive systems) @ FEL CTU 2021/2022

Authors: Tomáš Trejdl <trejdtom@fel.cvut.cz>, Vojtěch Tilhon <tilhovoj@fel.cvut.cz>

## Zadání
Vytvořte modul, který robotovi umožní kopírovat pohyby rukou demonstrované člověkem
Učení pomocí imitace je důležitou součástí kognitivního rozvoje u dětí. Co všechno potřebujeme znát a chápat pro správné zrcadlení gest? Vytvořte/najděte algoritmus, který z vidění dokáže získat pozice rukou člověka a naimplementujte modul pro robota Pepper, který mu umožní v reálném čase napodobovat lidská gesta.
Odkazy: https://viso.ai/deep-learning/pose-estimation-ultimate-overview/, https://en.wikipedia.org/wiki/Mirror_neuron
Doporučené nástroje, knihovny apod.: Python, naoqi API, existující již natrénované neuronové sítě, openpose

## Our solution

Use the `pytorch-openpose` library to estimate pose from Pepper's camera feed. Receive a list of keypoints of body and hands. Compute angles between arm and finger segments in a 2D plane. Offset computed angles into a range that Papper can move to. Tell Pepper to mode shoulders and elbows to specified angle.

Pepper-Controller only allows us to open / close hands, not move individuals fingers. We detect if a hand is open by the angle between finger segments. If the angle is large enough (over 150 degrees), we assume the hand is open and tell Pepper to open hand.

Files that containt our solution

- `/pepper-pose-mirroring/pytorch_openpose/mirror_pose.py`
- `/pepper-pose-mirroring/pytorch_openpose/src/angle_detection.py`

### Possible improvements

- Calibrate body segment lengths at the beginning and estimate is the person is rotated (arm closer to Pepper will be shorter in the 2D camera plane) and correct for it
- Detect open/closed hands by more fingers
- Detect if the person's forearm is facing up or down and rotate Pepper'' elbow accordingly
- Use Pepper's depth camera to estimate if the person's hand is in front of him/her

## Prerequisites

- python 3.5.6
- [conda](https://conda.io)

## Installation

```sh
git clone https://github.com/tomastrejdl/pepper-pose-mirroring.git

cd pepper-pose-mirroring

# Clone the Pepper-Controller repo
git clone https://github.com/incognite-lab/Pepper-Controller.git

# Add Pepper-Controller to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/Pepper-Controller/directory
```

Download [model files](https://github.com/Hzzone/pytorch-openpose#download-the-models) and put them in the `/pytorch-openpose/model` directory

```sh
conda env create -f environment.yml

conda activate pepper_env
```

## Run 

Update Pepper URL in `/pytorch-openpose/mirror_pose.py`

```sh
cd pytorch-openpose

python ./mirror_pose.py
```


## Implementation

Pepper Joints Documentation
- http://doc.aldebaran.com/2-0/family/juliette_technical/joints_juliette.html

### Joints we will be controlling

- Shoulder Roll - Left and Right
- Elbow Roll - Left and Right
- Hand - Open / Close

![](http://doc.aldebaran.com/2-0/_images/juliet_joints.png)
![](http://doc.aldebaran.com/2-0/_images/joint_right_arm.png)
![](http://doc.aldebaran.com/2-0/_images/joint_left_arm.png)