#!/usr/bin/env python
import socket
import select
import time
import rospy

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from unity_bridge.srv import SimulationMode, SimulationModeResponse
from unity_bridge.srv import SimulationStep, SimulationStepResponse
from unity_bridge.srv import SimulationReset, SimulationResetResponse

class TCP_Communicator:
    def __init__(self):
        # Initialize the ros node
        rospy.init_node('tcp_communication')

        # Create a TCP/IP socket for sending and receiving
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Wait till unity simulation opens the tcp socket to receive actions
        rospy.sleep(15)
        
        # Connect the sockets
        self.sock_recv.bind(('localhost', 9996))
        self.sock_recv.listen(1)
        self.sock_send.connect(('localhost', 9997))

        # Wait for Unity to connect
        self.sock_recv, self.client_address_recv = self.sock_recv.accept()
        
        # Publishers
        self.pub_current_state = rospy.Publisher('/current_state', Odometry, queue_size=10)
        
        # Flags
        self.training_mode = True
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.v = 0.0
        
        # Message
        self.odom_msg = Odometry()
        
        # Enable the rosservices
        self.SimulationStep_service = rospy.Service('SimulationMode', SimulationMode, self.SimulationMode)
        self.SimulationStep_service = rospy.Service('SimulationStep', SimulationStep, self.SimulationStep)
        self.SimulationStep_service = rospy.Service('SimulationReset', SimulationReset, self.SimulationReset)

        rospy.logwarn("TCP SERVER IS READY")
    
    def SimulationMode(self, simulation_mode):
        # Change the mode of the unity simulation
        if simulation_mode.sim_mode == "Train":
            self.sock_send.sendall("autoSimulation OFF;".encode())
            self.empty_socket()
            mode = "TRAINING"
        else:
            self.sock_send.sendall("autoSimulation ON;".encode())
            self.empty_socket()
            self.training_mode = False
            mode = "INFERENCE"
        
        return SimulationModeResponse(mode)

    def SimulationStep(self, action_command):
        # Set the propeller speed depending on the simulation mode
        w = action_command.action
        
        self.send_w_to_unity(w)
        self.x , self.y , self.z, self.v = self.communicate()
        
        return SimulationStepResponse(self.x , self.y , self.z, self.v)
    
    def SimulationReset(self, simulation_reset):
        # Set the simulation back to initial state
        if simulation_reset.reset_flg:
            self.sock_send.sendall("reset 0.0 0.0;".encode())
        
        self.x , self.y , self.z, self.v = self.communicate()
        
        return SimulationResetResponse(self.x , self.y , self.z, self.v)
    
    def send_w_to_unity(self, action_command):
        # Send the action command to the unity simulation
        w = action_command
        if self.training_mode:
            # Unity will wait for the action to update the state
            self.sock_send.sendall(f"step {w} {w} {w} {w};".encode())
        else:
            # Unity will run in real time
            self.sock_send.sendall(f"w {w} {w} {w} {w};".encode())
        
    def empty_socket(self):
        # Empty the socket to get the newest state messages
        input = [self.sock_recv]
        
        while 1:
            inputready, o, e = select.select(input,[],[], 0.0)
            if len(inputready) == 0: break
            for s in inputready: s.recv(1024)

    def communicate(self):
        # Receive the state from unity simulation via tcp
        state_msg = self.sock_recv.recv(1024).decode('utf-8')
        state_arr = state_msg.split(';')[-2].split(' ')

        # Publish the odometry message to /current_state topic
        self.odom_msg.pose.pose.position.x = float(state_arr[0])
        self.odom_msg.pose.pose.position.y = float(state_arr[2])
        self.odom_msg.pose.pose.position.z = float(state_arr[1])
        self.pub_current_state.publish(self.odom_msg)

        return float(state_arr[0]), float(state_arr[2]) , float(state_arr[1]), float(state_arr[3])

if __name__ == '__main__':
    try:
        # Instantiate the TCP Communication
        tcp_com = TCP_Communicator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
