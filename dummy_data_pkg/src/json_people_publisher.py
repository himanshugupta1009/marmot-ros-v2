#!/usr/bin/env python


#This ROS node reads a JSON file of precomputed pedestrian trajectories and
#  publishes the positions of all pedestrians at 100 Hz 
# in a single, synchronized message using the custom PeoplePoseArray.msg message type.

import rospy
import json
import os
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from dummy_data_pkg.msg import PeoplePoseArray

# === Config ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "..", "..", "MarmotSimData", "human_paths_testing.json") # Update as needed
PUBLISH_RATE = 100  # Hz

# Loads JSON data: an array of pedestrian each with an ID and a path of states
def load_trajectories(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    rospy.init_node("dummy_data_publisher", anonymous=False)
    rospy.loginfo("Starting dummy data publisher at 100Hz")

    data = load_trajectories(JSON_PATH)
    total_steps = max(len(p["path"]) for p in data)  # Determines the longest trajectory

    # Publisher sends PeoplePoseArray messages
    pub = rospy.Publisher("/car/dummy/people_poses", PeoplePoseArray, queue_size=10)

    rate = rospy.Rate(PUBLISH_RATE)
    t = 0

    while not rospy.is_shutdown() and t < total_steps:
        msg = PeoplePoseArray()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"

        # For each pedestrian, if they have data for the current time step, add it
        for ped in data:
            path = ped["path"]
            if t < len(path):
                state = path[t]
                pose = Pose()
                pose.position.x = state["x"]
                pose.position.y = state["y"]
                pose.position.z = 0.0
                pose.orientation.w = 1.0  # identity quaternion

                msg.poses.append(pose)
                msg.ids.append(ped["id"])

        pub.publish(msg)  # Publish a single synchronized message for all pedestrians
        t += 1
        rate.sleep()

    rospy.loginfo("Publishing complete")

if __name__ == "__main__":
    main()

