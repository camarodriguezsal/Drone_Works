#!/usr/bin/env python
import numpy as np
import gym
import json
import random
import math

from prompt_toolkit.key_binding.bindings.named_commands import end_kbd_macro
import rospy
import time
import socket
import select
import time
from gym import spaces
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from std_msgs.msg import Bool

from unity_bridge.srv import SimulationMode
from unity_bridge.srv import SimulationStep
from unity_bridge.srv import SimulationReset

MAX_PROPELLER_SPEED = 100.0
MIN_PROPELLER_SPEED = 0.0
MAX_VELOCITY = 30.0
MAX_ALTITUDE = 20.0
TARGET_ALTITUDE = 10.0
MAX_STEPS = 600

class QuadrotorTrainEnv(gym.Env):
    """ DRONE TRAINING ENVIRONMENT FOR OpenAI GYM """
      
    def __init__(self, simulation_mode = "Train"):
        """ INITIALIZE THE ENVIRONMENT """
        super(QuadrotorTrainEnv, self).__init__()

        # Initialize the parameters
        self.simulation_mode = simulation_mode  # flag to set the simulation mode
        self.reward = 0.0               # reward returned after each step 
        self.step_count = 0             # number of timesteps in an episode
        self.action_w = 0.0             # propeller speed send to unity
        self.reset_flag = False         # flag to set to initial state
        self.vel_z = -1.0               # vertical velocity [-1 ; 1]
        self.x_pos = 0.0                # x-axis position [m]
        self.y_pos = 0.0                # y-axis position [m]
        # z-axis position as numpy array [-1 ; 1]
        self.current_altitude = np.array([-1.0])  
        # z-axis target position as numpy array [-1 ; 1]
        self.target_altitude = self.normalize_altitude(np.array([TARGET_ALTITUDE])) 
        
        # Wait for the ros services 
        rospy.loginfo('Waiting for SimulationMode Service to become available.')
        rospy.wait_for_service('SimulationMode')
        rospy.loginfo('Waiting for SimulationStep Service to become available.')
        rospy.wait_for_service('SimulationStep')
        rospy.loginfo('Waiting for SimulationReset Service to become available.')
        rospy.wait_for_service('SimulationReset')

        # Instantiate ros service objects
        self.sim_mode = rospy.ServiceProxy('SimulationMode', SimulationMode)
        self.sim_step = rospy.ServiceProxy('SimulationStep', SimulationStep)
        self.sim_reset = rospy.ServiceProxy('SimulationReset', SimulationReset)

        self.action_space = spaces.Box(
            np.array([-1.0]), np.array([1.0]), shape=(1,), dtype=np.float32)
        # Continuous observation space for the altitude 
        self.observation_space = spaces.Box(
          low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32)

        # Set the simulation mode using the rosservice
        auto_simulation = self.sim_mode(self.simulation_mode)
        rospy.logwarn("%s MODE", auto_simulation.mode)

        rospy.logwarn("ENVIRONMENT READY")

    def get_reward(self):
        """ REWARD FUNCTION """

        reward = math.exp(-( (abs(self.current_altitude[0]-self.target_altitude[0]) * (MAX_ALTITUDE/2.0) )**0.5)) -0.05

        return reward 

    def step(self, propeller_speed):
        """ EXECUTE ONE TIME STEP IN THE GYM """
        # Propeller command mapping from [-1 ; 1] to [MIN_PROPELLER_SPEED ; MAX_PROPELLER_SPEED]
        self.action_w  = self.remap_action(propeller_speed)
        
        # Use the rosservice to get the current drone state via tcp communication
        drone_state = self.sim_step(self.action_w)
        self.current_altitude = self.normalize_altitude(np.array([drone_state.pos_z]))
        self.x_pos = drone_state.pos_x
        self.y_pos = drone_state.pos_y
        self.vel_z = drone_state.vel_z

        # Get the reward for the current step
        self.reward = self.get_reward()
     
        # Calculate the episode time
        info = {"time": (time.time() - self.start)}

        # Reset criteria based on number of steps
        if (self.step_count > MAX_STEPS) :
            #rospy.loginfo("MAX STEPS REACHED")
            done = True
        # Reset criteria based on maximum altitude
        elif (self.current_altitude[0] > (MAX_ALTITUDE / MAX_ALTITUDE)):
            #rospy.loginfo("MAX ALTITUDE REACHED")
            self.reward = -1
            done = True
        # Reset criteria based on lateral position error combined with low altitude
        elif abs(self.x_pos) > 0.2 or abs(self.y_pos) > 0.2 and self.current_altitude[0] < self.normalize_altitude(2.0):
            #rospy.loginfo("X - Y POSITION ERROR")
            self.reward = -1
            done = True
        # Reset criteria based on the bug in unity: no negative altitude
        elif self.current_altitude[0] < -1.0:
            #rospy.loginfo("NEGATIVE ALTITUDE")
            done = True
        else:
            done = False
            
        # Increment the step count
        self.step_count += 1
        
        # Return array for the observation [normalized altitude ;  normalized vertical velocity]
        obs_return = np.array([self.normalize_altitude(drone_state.pos_z) , self.normalize_velocity(drone_state.vel_z)])
    
        return obs_return, self.reward, done, info 

    def reset(self):
        """  RESET TO INITIAL STATE """
        # Reset the state of the environment to an initial state
        self.start = time.time()
        self.step_count = 0
        
        # Use the rosservice to get the reset drone state via tcp communication
        drone_state = self.sim_reset(True)

        self.reset_flag = False
        self.current_altitude = self.normalize_altitude(np.array([drone_state.pos_z]))
        self.x_pos = drone_state.pos_x
        self.y_pos = drone_state.pos_y
        self.vel_z = drone_state.vel_z
        
        obs_return = np.array([self.normalize_altitude(drone_state.pos_z) , self.normalize_velocity(drone_state.vel_z)])
        
        return obs_return
    
    def normalize_altitude(self, altitude):
        # Normalize the altitude into [0.0 ; MAX_ALTITUDE] to [-1 ; 1]
        return altitude / (MAX_ALTITUDE/2.0) - 1
    
    def normalize_velocity(self, velocity):
        # Normalize the velocity from [0.0 ; MAX_VELOCITY] to [-1 ; 1]
        return velocity / (MAX_VELOCITY/2.0) - 1  

    def remap_action(self, action):
        # Map the action command from [-1 ; 1] to [0.0 ; MAX_PROPELLER_SPEED]
        return (float(action[0]) + 1) * (MAX_PROPELLER_SPEED / 2.0) 
