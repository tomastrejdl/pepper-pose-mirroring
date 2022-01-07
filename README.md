# pepper-pose-mirroring
Semester project for KSY (Kognitive systems) @ FEL CTU 2021

# Zadání
Vytvořte modul, který robotovi umožní kopírovat pohyby rukou demonstrované člověkem
Učení pomocí imitace je důležitou součástí kognitivního rozvoje u dětí. Co všechno potřebujeme znát a chápat pro správné zrcadlení gest? Vytvořte/najděte algoritmus, který z vidění dokáže získat pozice rukou člověka a naimplementujte modul pro robota Pepper, který mu umožní v reálném čase napodobovat lidská gesta.
Odkazy: https://viso.ai/deep-learning/pose-estimation-ultimate-overview/, https://en.wikipedia.org/wiki/Mirror_neuron
Doporučené nástroje, knihovny apod.: Python, naoqi API, existující již natrénované neuronové sítě, openpose


## Prerequisites

- python
- [conda](https://conda.io)

## Installation

`git clone https://github.com/tomastrejdl/pepper-pose-mirroring.git`

`cd pepper-pose-mirroring`

`conda env create -f environment.yml`

`conda activate pepper_env`

## Run Mediapipe Demo

`python3 ./demo/mediapipe-holistic-webcam-demo.py`


## Implementation

Calculate angle between two edges
- https://stackoverflow.com/questions/69154914/calculating-angles-of-body-skeleton-in-video-using-openpose

Pepper Joints Documentation
- http://doc.aldebaran.com/2-0/family/juliette_technical/joints_juliette.html

### Joints we will be controlling

- Shoulder Roll - Left and Right
- Elbow Roll - Left and Right
- Hand - Open / Close

![](http://doc.aldebaran.com/2-0/_images/juliet_joints.png)
![](http://doc.aldebaran.com/2-0/_images/joint_right_arm.png)
![](http://doc.aldebaran.com/2-0/_images/joint_left_arm.png)