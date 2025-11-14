# SO-101 Controller

## Overview

Run a MuJoCo simulation and stream optionnaly joint angles to the real robot servos. Project includes a simple inverse-kinematics solver and a serial driver for ST3215 servos.

## Installation

Requires Python 3.10. Install dependancies with :

```sh
pip install -r requirements.txt
```

## Run

```sh
python main.py --simulation
```

## Project Organization

The project is devided in 3 parts :

- main.py : main program to launch with :

```sh
python main.py
```

or for MacOS :

```sh
mjpython main.py
```

- so101_driver : driver to control ST3215 motors (based on the ST3215 python package)

- ik.py : hard coded inverse kinematic class using MuJoCo data
