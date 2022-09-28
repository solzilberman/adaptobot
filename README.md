# Adaptive Rover

## High-level Overview
<img src="assets/archio.png" width="500px"/>

## Description
This simulation consists of 4 main components, a rover, an adaptive manager, a knowledge base, and an environment. 

### 1. Rover
The rover is designed to patrol waypoints iteratively while avoiding obstacles, maintaining appropriate battery levels, and staying on schedule. The rover has a *camera* able to detect objects within a specified range. If an obstacle is detected within this range, the rovers LEC is enabled. 

#### 1.1 Rover's LEC
The rovers LEC is a convolutional neural network trained on the CIFAR10 dataset. When the rover detects an obstacle within a specified range, it uses the CNN to classify the object. If the rover makes a correct prediction, the obstacle is *avoided* and moved to a new position. If the rover is unable to detect the obstacle, it enters manual mode and the user must use arrow keys to navigate it to the target waypoint. 

This flow for detecting objects is a "dummy" way to check if the LEC is degraded. This mimics the behavior oracle described in MoDALAS by ensuring the system is able to know when it encounters an obstacle it is not equipped to handle, and successively triggering an adaptation tactic.  

### 2. SAS (SelfAdaptiveSystem)
This system consists of a knowledge base and a given rover to manage. Following the MAPE-K loop, the SAS monitors the rovers environment and internal state by requesting various data from the rover (loosely emulates ROS topic requests).  

#### 2.1 Assurance Case provided by GSN
<img src="./gsn/gsn_img.jpg"/>
This model is parsed and utility functions extracted for later use by the SAS.

#### 2.2 Adaptation Tactics
These are designed after creating the GSN model and map utility violation patterns to actions that the rover can carry out.
```python
# SAS.py
...
tactics = {
    "11": [],
    "01": ['self.rover.MODE="charge"', "self.isUtilityViolatedBattery = False"],
    "10": ['self.rover.MODE="manual"', "self.isUtilityViolatedCV = False"],
    "00": [
        'self.rover.MODE="charge"',
        "self.isUtilityViolatedBattery = False",
        'self.rover.MODE="manual"',
        "self.isUtilityViolatedCV = False",
    ],
}
...
```
#### 2.3 CNN Model
This is trained and stored prior to deployment of the rover.

### 3. Environment
The environment is a custom simulated field hosted by pygame 2d graphics library. The environment consists of waypoints that the rover must reach, and obstacles for the rover to avoid. The rover has direct access to this environment. 
___
## Demo
*please note: The simulated environment is running on Win10 wsl2 and displayed through a local X11 server. This causes visual artifacts during simulation recording which are visible below but not present when running the simulation.*

![gif](https://user-images.githubusercontent.com/45021394/192883421-a2823bac-e09a-433f-9a7c-d807eafd1776.gif)

This demo shows some of the adaptive capabilities of the rover. When the battery utility function is violated, the rover takes the `solar_charge` tactic that causes it to switch to *charge* mode, stop moving, and charge until battery level is sufficient. The live demo has a real-time terminal based dashboard displaying various rover/system logs. 

___
## Todo
- [ ] better obstacle placement
- [ ] KAOS goal model

## Citations
- https://www.cs.toronto.edu/~kriz/cifar.html
- https://dl.acm.org/doi/10.1145/3365438.3410952
- https://www.pygame.org/docs/
- https://cse.msu.edu/~langfo37/gsndraw/
- https://cse.msu.edu/~langfo37/kaosdraw/
- https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
- https://ieeexplore.ieee.org/document/9592499 

### Assets
- [bg surface](https://www.google.com/https%3A%2F%2Fforum.flightgear.org%2Fviewtopic.php%3Ff%3D5%26t%3D37950)
