# human_aware_navigation_modifiedbyansh

Attempting to generate a simulation pipeline for identifying root cause of the timing mishandling.

## Quickstart — ROS Realtime (6 terminals)

> In **each** terminal, run:
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/catkin_ws/devel/setup.bash


T1 — roscore

roscore


T2 — set_action service

rosrun dummy_data_pkg set_action_service.jl


T3 — get_live_data service

rosrun dummy_data_pkg get_live_data_service.jl


T4 — realtime client (planner)

rosrun dummy_data_pkg realtime_vehicle_client.jl


T5 — realtime vehicle node

rosrun dummy_data_pkg realtime_vehicle_node.jl
# waits ~20 s for people data (Wait time can be changed)


T6 — JSON people publisher (start after T5 shows the wait message)

rosrun dummy_data_pkg json_people_publisher.py
# Change the publishing file name for different number of humans in environment