#!/usr/bin/env python
import rospy
import gym
import os
import numpy as np
import time
import datetime
from QuadrotorTrainEnv import QuadrotorTrainEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes, CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from trainer.srv import EnjoyStart

# Define the training parameters
MAX_EPISODES = 70                   # Number of maximum episodes to stop the training
BATCH_SIZE = 32                     # Batch size for the training
LEARNING_RATE = 0.001               # Learning rate for the training
LEARNING_STARTS = 400               # Use the first LEARNING_STARTS steps for warmin up
TOTAL_TIMESTEPS = int(5e4)          # Number of maximum timesteps to stop the training
TRAIN_FREQ = 1                      # Update the model after every TRAIN_FREQ timesteps
ENT_COEF = 0.1                      # Entropy regularization coefficient to control the exploration/exploitation trade-off.

# Define the callback parameters
SAVE_FREQ = int(3e3)                # Frequency to save the mid models after every SAVE_FREQ step
REWARD_THRESHOLD = 550              # Reward threshold to early stop the training
EVAL_FREQ = int(3e3)                # Evaluate the agent every eval_freq call of the callback
N_EVAL_EPISODES = int(2)            # The number of episodes to test the agent

# tensorboard --logdir ./catkin_ws/src/trainer/tensorboard_log_dir  

def train():
    # Get the directories
    model_save_dir = rospy.get_param(param_name='model_save_path')              # directory to save the trained model
    tensorboard_log_dir = rospy.get_param(param_name='tensorboard_log_path')    # directory to save the tensorboard logs
    checkpoint_log_dir = rospy.get_param(param_name='checkpoint_log_path')      # directory to save the model at every SAVE_FREQ steps
    eval_log_dir = rospy.get_param(param_name='eval_log_path')                  # directory to save the evaluated model
    
    # Instantiate the environment
    env = QuadrotorTrainEnv(simulation_mode = "Train")

    # Evaluation environment for early stop criteria
    eval_env = QuadrotorTrainEnv(simulation_mode = "Train")

    # Actor (aka pi) and the critic (Q-function aka qf)
    policy_kwargs = dict(net_arch=dict(pi=[32, 32], qf=[32, 32]))
    
    # SAC 
    model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, 
        learning_starts=LEARNING_STARTS, tensorboard_log=tensorboard_log_dir, train_freq = TRAIN_FREQ, ent_coef=ENT_COEF)
    
    '''
    # DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, verbose=1, action_noise = action_noise, policy_kwargs=policy_kwargs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, 
    learning_starts=LEARNING_STARTS, tensorboard_log=tensorboard_log_dir, train_freq = TRAIN_FREQ)
    '''

    rospy.logwarn("MODEL CREATED, STARTING TRAINING")

    # Training callbacks
    # Stop the training after certain number of episodes
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=MAX_EPISODES, verbose=1) 
    # Saving the model periodically 
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=checkpoint_log_dir, name_prefix='rl_quadrotor_midmodel')
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1 , 
        deterministic=True,  warn=False, eval_freq=EVAL_FREQ , n_eval_episodes=N_EVAL_EPISODES)
    # Create the callback list
    callback = CallbackList([callback_max_episodes, checkpoint_callback, eval_callback])

    # Timer to get the training time
    start_time = time.time()

    # Start training
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback = callback)    
    training_duration = time.time() - start_time

    rospy.logwarn("FINISHED TRAINING")
    rospy.logwarn("DEPLOYMENT + EVALUATION + TRAINING DURATION: %s", datetime.timedelta(seconds=training_duration))

    # Save the model
    model.save(model_save_dir)
    
    rospy.logwarn("MODEL SAVED, READY TO DEPLOY")

    # Start the inference mode by calling the service
    rospy.loginfo('Waiting for EnjoyStart Service to become available.')
    rospy.wait_for_service('EnjoyStart')
    enjoy_start_srv = rospy.ServiceProxy('EnjoyStart', EnjoyStart)
    finished = enjoy_start_srv(True)

if __name__ == '__main__':
    try:
        rospy.init_node('trainer')
        train()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass    
