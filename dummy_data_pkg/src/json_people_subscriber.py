#!/usr/bin/env python

#This ROS node subscribes to the /car/dummy/people_poses topic and 
# logs the synchronized positions of all pedestrians. 
# It's designed to process each time step as a consistent snapshot of the simulation.

import rospy, time
from dummy_data_pkg.msg import PeoplePoseArray

_start_time = None
ped_pos_dict = {}

# This callback is called once per time step
def callback(msg):
    global _start_time, ped_pos_dict
    # on first message, capture the start time
    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time
    rospy.loginfo("Received {} poses at {:.3f}s since start".format(len(msg.ids), elapsed))

    # Iterate over all poses and IDs in the message
    for i in range(len(msg.ids)):
        ped_id = msg.ids[i]
        pose = msg.poses[i].position
        rospy.loginfo("Ped {} -> x: {:.2f}, y: {:.2f}".format(ped_id, pose.x, pose.y))
        ped_pos_dict.setdefault(ped_id, []).append((pose.x, pose.y))

def main():
    rospy.init_node("json_people_subscriber")

    # Subscribe to the synchronized pedestrian position topic
    rospy.Subscriber("/car/dummy/people_poses", PeoplePoseArray, callback)
    rospy.loginfo("Subscribed to /car/dummy/people_poses")
    rospy.spin() # Keep node alive and listening for messages

if __name__ == "__main__":
    main()
