# RL-Quadrotor

This is a document to introduce how to use this repository.

## Table of Contents

- [Environment](#environment)
- [Conda Preparation](#conda preparation)
- [Install](#install)
- [Start](#start)
- [Result](#result)

## Environment
- ### Ubuntu 18.04.5 LTS
- ### ROS melodic
- ### Python 3.8.10

## Conda Preparation

`conda create -n rl-quadrotor python=3.8 -y`

`conda activate rl-quadrotor`

`pip install -r requirements.txt`

## Install
- ### Unity Environment 
The linux build of the unity environment is stored here https://www.dropbox.com/s/cupkh5xub42e1b2/Linux_build_Data.zip?dl=0
catkin build command downloads the unity environment using the link and puts in the devel folder.

- ### ROS Structure
`git clone git@gitlab.lrz.de:ga42say/rl-quadrotor.git`

`cd catkin_ws`

`catkin build`

## Start
`source devel/setup.bash`

`roslaunch orchestrator run.launch`

### Tensorboard
`cd ..`

`tensorboard --logdir ./catkin_ws/src/trainer/tensorboard_log_dir`

## Result
### Video: 

