#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from state_updater_pkg.srv import UpdateState, UpdateStateResponse

import copy
import math
from scipy.spatial.transform import Rotation
from numpy import linalg as la
from datetime import datetime
import csv
import logging
import os
import time

# File Overview:
# - subscribes to Pose topic for each object tracked by Vicon
# - each subscriber has a callback function that stashes received pose in a global variable
# - controller requests state vector periodically by querying the UpdateState service
# - service returns the most recently collected poses from the Vicon topics

# Vicon state vector:
#   [Marmot x;
#   Marmot y;
#   Marmot theta;
#   Marmot v;
#   Ped1 x;
#   Ped1 y;
#   Ped2 x;
#   ...]
# - length = 4 + 2*n_peds

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class StateUpdater:


    # A class to manage the state of a vehicle and multiple pedestrians based on Vicon data.

    # This class subscribes to Pose and Twist topics for a vehicle and multiple pedestrians,
    # stores their current states, and provides a service to update and retrieve the state vector.
    # The state vector includes the vehicle's position, orientation, and velocity, as well as
    # the positions of the pedestrians.

    # Attributes:
    #     num_pedestrians (int): The number of pedestrians to track.
    #     current_veh_msg (PoseStamped): The latest Pose message for the vehicle.
    #     current_veh_twist_msg (TwistStamped): The latest Twist message for the vehicle.
    #     current_ped_msgs (list): List of the latest Pose messages for each pedestrian.
    #     hist_veh_msg (list): History of vehicle Pose messages.
    #     hist_ped_msgs (list): List of histories for each pedestrian's Pose messages.
    #     record_hist (bool): Flag to indicate if history recording is active.
    #     saved_hist (bool): Flag to indicate if history has been saved.


    def __init__(self, num_pedestrians):
        try:
            #subscribe to Vicon topic for each object

            # Store the number of pedestrians
            self.num_pedestrians = num_pedestrians
            
            # Initialize vehicle data
            self.current_veh_msg = None
            self.current_veh_twist_msg = None
            self.hist_veh_msg = []
            self.hist_veh_twist_msg = []
            
            # Initialize pedestrian data containers
            self.ped_subs = []            # List of pedestrian subscribers
            self.current_ped_msgs = [None] * num_pedestrians  # Latest message for each pedestrian
            self.hist_ped_msgs = [[] for _ in range(num_pedestrians)]  # History for each pedestrian
            
            self.record_hist = False
            self.saved_hist = False

            # Vehicle subscriber
            self.vrpn_sub_marmot_pose = rospy.Subscriber(
                "/car/vrpn_client_ros/vrpn_client_node/ADCL_Marmot/pose", 
                PoseStamped,
                callback=self.store_current_msg,
                callback_args='vehicle_pose',
                queue_size=1)

            # Vehicle twist subscriber
            self.vrpn_sub_marmot_twist = rospy.Subscriber(
                "/car/vrpn_client_ros/vrpn_client_node/ADCL_Marmot/twist", 
                TwistStamped,
                callback=self.store_current_msg,
                callback_args='vehicle_twist',
                queue_size=1)
            
            # Pedestrian subscribers
            for i in range(self.num_pedestrians):
                topic_name = "/car/vrpn_client_ros/vrpn_client_node/ADCL_Ped{}/pose".format(i+1)
                sub = rospy.Subscriber(
                    topic_name, 
                    PoseStamped,
                    callback=self.store_current_msg,
                    callback_args=i,
                    queue_size=1)
                self.ped_subs.append(sub)

            # Service for state estimation
            self.update_state_srv = rospy.Service(
                "/car/state_updater/get_state_update",
                UpdateState,
                self.update_state)
            
        except Exception as e:
            rospy.logerr("Failed to initialize StateUpdater: %s", str(e))    
        



        # subscriber callback function
        # Callback function to store the current Pose message for the vehicle or pedestrians.

        # Args:
        #     pose_msg (PoseStamped): The Pose message received from the subscriber.
        #     obj_id (str or int): Identifier for the object type (vehicle or pedestrian index).
     
    def store_current_msg(self, pose_msg, obj_id):
        logging.debug("store_current_msg called with obj_id=%s", obj_id)

        if isinstance(obj_id, str):  # Handle vehicle messages
            if obj_id == 'vehicle_pose':
                self.current_veh_msg = copy.deepcopy(pose_msg)
                if self.record_hist:
                    self.hist_veh_msg.append(pose_msg)
            elif obj_id == 'vehicle_twist':
                self.current_veh_twist_msg = copy.deepcopy(pose_msg)
                if self.record_hist:
                    self.hist_veh_twist_msg.append(pose_msg)
            else:
                rospy.logwarn("Unknown obj_id string: %s (type=%s)", obj_id, type(obj_id))

        elif isinstance(obj_id, int):  # Handle pedestrian messages
            if 0 <= obj_id < self.num_pedestrians:  # Ensure valid index
                self.current_ped_msgs[obj_id] = copy.deepcopy(pose_msg)
                if self.record_hist:
                    self.hist_ped_msgs[obj_id].append(pose_msg)
            else:
                rospy.logwarn("Invalid pedestrian index: %s (type=%s)", obj_id, type(obj_id))
        else:
            rospy.logwarn("Unexpected obj_id type: %s (type=%s)", obj_id, type(obj_id))





    # function called by service

    #     Service callback to update and return the current state vector.

    #     Args:
    #         req (UpdateState): The service request containing the record flag.

    #     Returns:
    #         UpdateStateResponse: The response containing the current state vector and pedestrian presence.
    
    def update_state(self, req):
        self.record_hist = req.record

      # data logging for times when recording has stopped and data needs saving
        if self.record_hist == False and self.saved_hist == False and len(self.hist_veh_msg) > 0: 
            self.save_s_hist()

        # Calculate length of state vector based on number of pedestrians
        current_state = [0] * (4 + 2 * self.num_pedestrians)
        peds_in_env = [False] * self.num_pedestrians
        
        # Vehicle state
        if self.current_veh_msg and self.current_veh_twist_msg:
            current_state[0:4] = self.convert_veh_state(self.current_veh_msg, self.current_veh_twist_msg)

        # Pedestrian states
        for i, ped_msg in enumerate(self.current_ped_msgs):
            if ped_msg:
                ped_state, ped_in_env = self.convert_ped_state(ped_msg)
                current_state[4 + 2 * i:6 + 2 * i] = ped_state
                peds_in_env[i] = ped_in_env  # Track whether the pedestrian is in the environment
        
        return UpdateStateResponse(state=current_state, peds_in_env=peds_in_env)




    # converts ROS PoseStamped message to regular (x,y,theta,v) variables
        # Converts ROS PoseStamped and TwistStamped messages to a vehicle state vector.

        # Args:
        #     pose_msg (PoseStamped): The Pose message for the vehicle.
        #     twist_msg (TwistStamped): The Twist message for the vehicle.

        # Returns:
        #     state: A list containing the vehicle's x, y, theta, and velocity

    def convert_veh_state(self, pose_msg, twist_msg):
        ori = pose_msg.pose.orientation
        rot = Rotation.from_quat([ori.x, ori.y, ori.z, ori.w])
        rot_euler = rot.as_euler('xyz', degrees=False)
        theta = rot_euler[2]

    
        y_cal = -0.162  # m from Vicon model center to rear axis
        x = pose_msg.pose.position.x + y_cal*math.cos(theta)
        y = pose_msg.pose.position.y + y_cal*math.sin(theta)

        v = la.norm([twist_msg.twist.linear.x, twist_msg.twist.linear.y])

        state = [x, y, theta, v]

        return state
    


# Converts a ROS PoseStamped message to a pedestrian state vector.

#         Args:
#             pose_msg (PoseStamped): The Pose message for the pedestrian.

#         Returns:
#             state: The pedestrian's state vector and a boolean indicating if the pedestrian is in the environment.
    def convert_ped_state(self, pose_msg):
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y

        state = [x, y]

        # TO-DO: come up with conditon to check if Vicon reads ped within environment
        #   - may read spare helmet outside workspace in storage area, or may return 0,0 or something if not visible at all
        ped_in_env = (x != 0 or y != 0)

        return state, ped_in_env
    


    # saves full Vicon history to csv file, called once at end of execution
    # TO-DO: manage histories from all objects
    #   - histories not necessarily same length, same time steps (hmm...)
    #   - see what each length is, might be close enough to not cause issues
    #   - probably easiest to save in separate files, pull together in analysis script



    def save_s_hist(self):
        # Saves the history of vehicle and pedestrian states to CSV files.
        # This function is called once at the end of execution to store the recorded history.

            timestamp = datetime.now().isoformat()
            base_dir = "/home/adcl/catkin_ws/src/marmot-ros-v2/controller_pkg/histories/"
            dir_name = os.path.join(base_dir, timestamp)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name) #Only creates the directory if it doesn't exist


            #vehicles history
            veh_filename = os.path.join(dir_name, "veh_hist_vrpn.csv")
            ped_filename = [os.path.join(dir_name, "ped{}_hist_vrpn.csv".format(i+1)) for i in range(self.num_pedestrians)]

            with open(veh_filename, 'w') as veh_file:
                writer = csv.writer(veh_file)
                # Write header row
                writer.writerow(['timestamp', 'x', 'y', 'theta', 'v'])

                veh_hist_len = min(len(self.hist_veh_msg), len(self.hist_veh_twist_msg))
            
                for k in range(veh_hist_len):

                        pose_msg_k = self.hist_veh_msg[k]
                        twist_msg_k = self.hist_veh_twist_msg[k]

                        timestamp = pose_msg_k.header.stamp.to_sec()
                        formatted_timestamp = datetime.fromtimestamp(timestamp).isoformat()

                        s_k = self.convert_veh_state(pose_msg_k, twist_msg_k)  # Pass both arguments
                        writer.writerow([formatted_timestamp] + s_k)

            # pedestrian histories
            for i,ped_filename in enumerate(ped_filename):
                with open(ped_filename, 'w') as ped_file:
                    writer = csv.writer(ped_file)
                    writer.writerow(['timestamp', 'x', 'y'])
                    for msg in self.hist_ped_msgs[i]:
                            timestamp = msg.header.stamp.to_sec()
                            formatted_timestamp = datetime.fromtimestamp(timestamp).isoformat()
                            writer.writerow([formatted_timestamp] + self.convert_ped_state(msg)[0])

            self.saved_hist = True

            print("Save complete for vehicle and pedestrian histories")
            return




if __name__ == '__main__':
    try:
        rospy.init_node("state_updater")

        # Get the number of pedestrians from a ROS parameter, defaulting to 4 if not set. To control this value, edit state_estimator.launch file
        num_pedestrians = rospy.get_param("~num_pedestrians", 4)
        
        # Initialize the StateEstimator with the specified number of pedestrians
        state_updater = StateUpdater(num_pedestrians)

        print("\n--- hello from state_updater.py ---\n")

        rospy.spin()
    except rospy.ROSInterruptException:
        pass