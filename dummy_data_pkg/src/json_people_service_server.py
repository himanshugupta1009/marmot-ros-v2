#!/usr/bin/env python

import rospy, json, os
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from dummy_data_pkg.msg import PeoplePoseArray
from dummy_data_pkg.srv import GetPeople, GetPeopleResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "..", "..", "MarmotSimData", "human_paths_testing.json")
data = None
total_steps = 0
step_counter = 0

def load_trajectories():
    global data, total_steps
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    total_steps = max(len(p["path"]) for p in data)

def handle_get_people(req):
    global step_counter, data, total_steps
    msg = PeoplePoseArray()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"

    for ped in data:
        path = ped["path"]
        if step_counter < len(path):
            state = path[step_counter]
            pose = Pose()
            pose.position.x = state["x"]
            pose.position.y = state["y"]
            pose.orientation.w = 1.0
            msg.poses.append(pose)
            msg.ids.append(ped["id"])
    step_counter += 1
    return GetPeopleResponse(people=msg)

def main():
    rospy.init_node("json_people_service_server")
    rospy.loginfo("Starting pedestrian trajectory service server...")
    load_trajectories()
    s = rospy.Service("get_people_snapshot", GetPeople, handle_get_people)
    rospy.spin()

if __name__ == "__main__":
    main()
