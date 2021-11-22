#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


TARGET_ALTITUDE = 10.0
FREQUENCY = 20.0 

class Visualiser:
    def __init__(self):
        #plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'r' , lw=2)
        self.ax.grid()
        self.ax.set(xlabel='time (s)', ylabel='position error (m)', title='Position Error over Time')
        self.x_data, self.y_data = [] , []

    def plot_init(self):
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-12, 12)
        #self.ln.set_data(x_data, y_data)
        return self.ln

    def odom_callback(self, msg):
        position_error = self.get_position_error(msg.pose.pose)
        self.y_data.append(position_error)
        # divided by 6 to get real-time indexing, /current_state publish rate is 20 Hz
        x_index = len(self.x_data)/FREQUENCY
        self.x_data.append(x_index+1/FREQUENCY)
    
    def get_position_error(self, pose):
        target_altitude = TARGET_ALTITUDE
        position_error = target_altitude - pose.position.z
        return -position_error   
    
    def update_plot(self, frame):
        xmin, xmax = self.ax.get_xlim()

        # extend the limits of the figure
        if len(self.x_data)/FREQUENCY >= xmax:
            self.ax.set_xlim(xmin, 2*xmax)
            self.ax.figure.canvas.draw()

        self.ln.set_data(self.x_data, self.y_data)

        return self.ln    

if __name__ == '__main__':
    #initialize the ros node
    rospy.init_node('error_listener')

    # instantiate the object
    vis = Visualiser()
    
    # subscribe to the topic
    sub = rospy.Subscriber('/current_state', Odometry, vis.odom_callback)
    
    # use the FuncAnimation from matplotlib to create real-time animation
    ani = FuncAnimation(vis.fig, vis.update_plot, init_func=vis.plot_init, interval=10)
    plt.show(block=True)

