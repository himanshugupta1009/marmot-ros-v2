#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from controller_pkg.srv import UpdateAction, UpdateActionRequest, UpdateActionResponse

# TO-DO: what is dt required to have smooth actions fron VESC?
#   - i.e., how often does speed need to be sent to VESC to continuously drive?

class ActionUpdater:
    global latest_a_msg

    def __init__(self):
        global latest_a_msg

        latest_a_msg = UpdateActionRequest()
        latest_a_msg.a_v = 0.0
        latest_a_msg.a_phi = 0.0

        self.update_action_srv = rospy.Service(
            "/car/controller/send_action_update",
            UpdateAction,
            self.store_a_msg)

        self.ackermann_pub = rospy.Publisher(
            "/car/mux/ackermann_cmd_mux/input/controller", 
            AckermannDriveStamped, 
            queue_size=2)

        dt = 0.05   # 20 Hz
        rate = rospy.Rate(1/dt)
        while True:
            self.publish_to_ackermann(latest_a_msg)
            rate.sleep()

        return

    def store_a_msg(self, a_req):
        global latest_a_msg
        latest_a_msg = a_req

        return UpdateActionResponse(True)

    def publish_to_ackermann(self, a_msg): 
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = rospy.Time.now()

        ackermann_msg.drive.speed = a_msg.a_v
        ackermann_msg.drive.steering_angle = a_msg.a_phi

        self.ackermann_pub.publish(ackermann_msg)

        return 


if __name__ == '__main__':
    try:
        rospy.init_node("action_updater")
        ActionUpdater()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass