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

class StateUpdater:
    current_veh_pose_msg = []
    current_veh_twist_msg = []
    current_ped1_msg = []
    current_ped2_msg = []
    current_ped3_msg = []
    current_ped4_msg = []

    record_hist = False
    saved_hist = False

    hist_veh_pose_msg = []
    hist_veh_twist_msg = []
    hist_ped1_msg = []
    hist_ped2_msg = []
    hist_ped3_msg = []
    hist_ped4_msg = []

    def __init__(self):
        # subscribe to Vicon topic for each object
        # - vehicle
        self.vrpn_sub_marmot_pose = rospy.Subscriber(
            "/car/vrpn_client_ros/vrpn_client_node/ADCL_Ped3/pose", 
            PoseStamped,
            callback=self.store_current_msg,
            callback_args=0,
            queue_size=1)

        self.vrpn_sub_marmot_twist = rospy.Subscriber(
            "/car/vrpn_client_ros/vrpn_client_node/ADCL_Ped3/twist", 
            TwistStamped,
            callback=self.store_current_msg,
            callback_args=1,
            queue_size=1)

        # - pedestrians
        self.vrpn_sub_ped1_pose = rospy.Subscriber(
            "/car/vrpn_client_ros/vrpn_client_node/ADCL_Ped1/pose", 
            PoseStamped,
            callback=self.store_current_msg,
            callback_args=2,
            queue_size=1)

        self.vrpn_sub_ped2_pose = rospy.Subscriber(
            "/car/vrpn_client_ros/vrpn_client_node/ADCL_Ped2/pose", 
            PoseStamped,
            callback=self.store_current_msg,
            callback_args=3,
            queue_size=1)

        self.vrpn_sub_ped3_pose = rospy.Subscriber(
            "/car/vrpn_client_ros/vrpn_client_node/ADCL_Ped3/pose", 
            PoseStamped,
            callback=self.store_current_msg,
            callback_args=4,
            queue_size=1)

        self.vrpn_sub_ped4_pose = rospy.Subscriber(
            "/car/vrpn_client_ros/vrpn_client_node/ADCL_Ped4/pose", 
            PoseStamped,
            callback=self.store_current_msg,
            callback_args=5,
            queue_size=1)

        # establish state_updater service as provider
        self.update_state_srv = rospy.Service(
            "/car/state_updater/get_state_update",
            UpdateState,
            self.update_state)

        return

    # subscriber callback function
    def store_current_msg(self, msg, vrpn_object):
        # - vehicle
        if vrpn_object == 0:
            self.current_veh_pose_msg = copy.deepcopy(msg)
            if self.record_hist == True:
                self.hist_veh_pose_msg.append(msg)

        elif vrpn_object == 1:
            self.current_veh_twist_msg = copy.deepcopy(msg)
            if self.record_hist == True:
                self.hist_veh_twist_msg.append(msg)

        # - pedestrians
        elif vrpn_object == 2:
            self.current_ped1_msg = copy.deepcopy(msg)
            if self.record_hist == True:
                self.hist_ped1_msg.append(msg)

        elif vrpn_object == 3:
            self.current_ped2_msg = copy.deepcopy(msg)
            if self.record_hist == True:
                self.hist_ped2_msg.append(msg)

        elif vrpn_object == 4:
            self.current_ped3_msg = copy.deepcopy(msg)
            if self.record_hist == True:
                self.hist_ped3_msg.append(msg)
                
        elif vrpn_object == 5:
            self.current_ped4_msg = copy.deepcopy(msg)
            if self.record_hist == True:
                self.hist_ped4_msg.append(msg)
        
        return

    # function called by service
    def update_state(self, req):
        self.record_hist = req.record

        if self.record_hist == False and self.saved_hist == False and len(self.hist_veh_msg) > 0: 
            self.save_s_hist()

        current_state = [0]*(4 + 2*(0))
        peds_in_env = [False]*4

        # - vehicle
        current_state[0:4] = self.convert_veh_state(self.current_veh_pose_msg, self.current_veh_twist_msg)

        # - pedestrians
        current_state[4:6], peds_in_env[0] = self.convert_ped_state(self.current_ped1_msg)
        current_state[6:8], peds_in_env[1] = self.convert_ped_state(self.current_ped2_msg)
        current_state[8:10], peds_in_env[2] = self.convert_ped_state(self.current_ped3_msg)
        current_state[10:12], peds_in_env[3] = self.convert_ped_state(self.current_ped4_msg)

        return UpdateStateResponse(current_state, peds_in_env)

    # converts ROS PoseStamped message to regular (x,y,theta,v) variables
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

    def convert_ped_state(self, pose_msg):
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y

        state = [x, y]

        # TO-DO: come up with conditon to check if Vicon reads ped within environment
        #   - may read spare helmet outside workspace in storage area, or may return 0,0 or something if not visible at all
        if True:
            ped_in_env = True
        else:
            ped_in_env = False

        return state, ped_in_env

    # saves full Vicon history to csv file, called once at end of execution
    # TO-DO: manage histories from all objects
    #   - histories not necessarily same length, same time steps (hmm...)
    #   - see what each length is, might be close enough to not cause issues
    #   - probably easiest to save in separate files, pull together in analysis script
    def save_s_hist(self):
        # veh_secs = datetime.fromtimestamp(self.current_veh_msg.header.stamp.secs)  
        # veh_nsecs = datetime.fromtimestamp(self.current_veh_msg.header.stamp.nsecs)
        # ped1_secs = datetime.fromtimestamp(self.current_ped1_msg.header.stamp.secs)  
        # ped1_nsecs = datetime.fromtimestamp(self.current_ped1_msg.header.stamp.nsecs)

        # print(veh_secs)
        # print(veh_nsecs)
        # print(ped1_secs)
        # print(ped1_nsecs)

        # TO-DO: add datetime to file name
        # TO-DO: add time stamp to each entry

        hist_path = "/home/adcl/catkin_ws/src/marmot-ros-v2/controller_pkg/histories/"

        # veh history
        f = open(hist_path + "veh_hist_vrpn.csv", 'w')
        writer = csv.writer(f)

        veh_hist_len = min(len(self.hist_veh_pose_msg), len(self.hist_veh_twist_msg))
        
        for k in range(0, veh_hist_len):
            pose_msg_k = self.hist_veh_pose_msg[k]
            twist_msg_k = self.hist_veh_twist_msg[k]

            s_k = self.convert_veh_state(pose_msg_k, twist_msg_k)

            writer.writerow(s_k)
        f.close()

        # ped1 history
        f = open(hist_path + "ped1_hist_vrpn.csv", 'w')
        writer = csv.writer(f)
        for msg_k in self.hist_ped1_msg:
            s_k = self.convert_ped_state(msg_k)
            writer.writerow(s_k)
        f.close()
        
        self.saved_hist = True
        print("save complete")

        return

if __name__ == '__main__':
    try:
        rospy.init_node("state_updater")

        print("\n--- hello from state_updater.py ---\n")

        StateUpdater()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass