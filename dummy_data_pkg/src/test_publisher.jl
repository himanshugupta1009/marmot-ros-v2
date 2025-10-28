#!/usr/bin/env julia

using RobotOS
@rosimport std_msgs.msg: String
rostypegen()
using .std_msgs.msg: StringMsg
# using RobotOS: NodeHandle

RobotOS.init_node("test_node")

pub = Publisher("/test", StringMsg; queue_size = 10, latch = true)

msg = StringMsg()
msg.data = "Hello world!"
publish(pub, msg)

RobotOS.spin()