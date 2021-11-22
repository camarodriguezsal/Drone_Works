#!/usr/bin/env python
import rospy
import gym
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from QuadrotorTrainEnv import QuadrotorTrainEnv
from trainer.srv import EnjoyStart, EnjoyStartResponse

INFERENCE_TIME = 15.0

def enjoy():
    # Get the directory
    model_save_dir = rospy.get_param(param_name='model_save_path')

    # Load the model
    #loaded_model = DDPG.load(model_save_dir)
    loaded_model = SAC.load(model_save_dir)
    
    # Instantiate the environment
    env = QuadrotorTrainEnv(simulation_mode = "Inference")

    # Set the rate of the ros node
    rate = rospy.Rate(20)

    # Start inference
    rospy.logwarn("STARTING THE INFERENCE")
    rospy.sleep(15)
    obs = env.reset()
    while not rospy.is_shutdown():
      action, _states = loaded_model.predict(obs, deterministic=True)
      obs, rewards, done, info = env.step(action)
      if done:
        obs = env.reset()
      elif info['time'] >= INFERENCE_TIME:
        rospy.logwarn("TIME UP")
        obs = env.reset()
      rate.sleep()  

def EnjoyStart_srv(enjoy_start):
    # Start the inference mode by using the enjoy script
    if enjoy_start.start:
        enjoy()

    return EnjoyStartResponse(True)

if __name__ == '__main__':
    try:
        rospy.init_node('inference')
        # enable the inference service
        s = rospy.Service('EnjoyStart', EnjoyStart, EnjoyStart_srv)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

