#!/usr/bin/env julia

using RobotOS
using BSON: @load, @save
using Dates

println("\n--- hello from controller.jl ---\n")

algs_path = "/home/adcl/Documents/marmot-algs/"
# include(algs_path * "HJB-planner/HJB_generator_functions.jl")
# include(algs_path * "HJB-planner/HJB_planner_functions.jl")

HAN_path = "/home/adcl/Documents/human_aware_navigation/src/"
# include(HAN_path * "main.jl")
# include(HAN_path * "utils.jl")
# include(HAN_path * "pomdp_planning.jl")
# include(HAN_path * "belief_tracker.jl")
# include(HAN_path * "main.jl")

# ROS connections:
#   - to state_updater node as CLIENT to state_updater service
#   - to ack_publisher node as CLIENT of ack_publisher service

@rosimport state_updater_pkg.srv: UpdateState
@rosimport belief_updater_pkg.srv: UpdateBelief
@rosimport controller_pkg.srv: UpdateAction

rostypegen()
using .state_updater_pkg.srv
using .belief_updater_pkg.srv
using .controller_pkg.srv

# (?): these 3 lines are duplicated, can delete?
# discount(p::POMDP_Planner_2D_action_space) = p.discount_factor
# isterminal(::POMDP_Planner_2D_action_space, s::POMDP_state_2D_action_space) = is_terminal_state_pomdp_planning(s,location(-100.0,-100.0));
# actions(m::POMDP_Planner_2D_action_space,b) = get_actions_non_holonomic(b)

# call state_updater service as client
function state_updater_client(record)
    wait_for_service("/car/state_updater/get_state_update")
    update_state_srv = ServiceProxy{UpdateState}("/car/state_updater/get_state_update")

    resp = update_state_srv(UpdateStateRequest(record))

    return resp.state
end

# call state_updater service as client
function belief_updater_client()
    wait_for_service("/car/belief_updater/get_belief_update")
    update_belief_srv = ServiceProxy{UpdateBelief}("/car/belief_updater/get_belief_update")

    resp = update_belief_srv(UpdateBeliefRequest(true))

    return resp.belief_array
end

function ros2pomdp_belief(belief_ros)
    l = length(belief_ros)
    belief = Array{belief_over_human_goals,1}()
    for i in 1:4:l
        b = belief_over_human_goals(belief_ros[i:i+3])
        push!(belief,b)
    end
    return belief
end

# call ack_publisher service as client
function action_updater_client(a)
    wait_for_service("/car/controller/send_action_update")
    update_action_srv = ServiceProxy{AckPub}("/car/controller/send_action_update")

    a_req = UpdateActionRequest(a[1], a[2])
    resp = update_action_srv(a_req)
end

function pomdp2ros_action(a_pomdp, v_kn1, Dt, veh_L)
    # define limits
    v_min = 0.0
    v_max = 1.0
    phi_max = 0.475

    # calculate velocity
    Dv = a_pomdp[2]
    v_k = v_kn1 + Dv
    clamp!(v_k, v_min, v_max)

    # calculate steering angle
    arc_length = v_k*Dt
    if(arc_length != 0)
        phi_k = atan(a_pomdp[1] * veh_L/arc_length)
        clamp!(phi_k, -phi_max, phi_max)
    else
        phi_k = 0.0
    end

    return [v_k, phi_k]
end

function main()
    # initialize ROS controller node
    init_node("controller")

    return

    # initialize utilities
    end_run = false
    o_hist = []
    a_hist = []

    max_plan_steps = 2*60*4
    planning_Dt = 0.5
    planning_rate = Rate(1/planning_Dt)

    plan_step = 1

    input = aspen
    exp_details, pomdp_details, output = get_details_from_input_parameters(input)
    env = generate_environment(input.env_length,input.env_breadth,input.obstacles)
    exp_details.env = env
    exp_details.human_goal_locations = get_human_goals(env)
    veh = Vehicle(input.veh_start_x, input.veh_start_y, input.veh_start_theta, input.veh_start_v)
    veh_sensor_data = vehicle_sensor(human_state[],Int64[],belief_over_human_goals[])
    veh_goal = location(input.veh_goal_x,input.veh_goal_y)
    veh_params = es_vehicle_parameters(input.veh_L,input.veh_max_speed,veh_goal)
    env_humans, env_humans_params = generate_humans(env,veh,exp_details.human_start_v,exp_details.human_goal_locations,exp_details.num_humans_env,exp_details.user_defined_rng)
    initial_sim_obj = simulator(env,veh,veh_params,veh_sensor_data,env_humans,env_humans_params,exp_details.simulator_time_step)

    #Define POMDP, POMDP Solver and POMDP Planner
    extended_space_pomdp = extended_space_POMDP_planner(pomdp_details.discount_factor,pomdp_details.min_safe_distance_from_human,
                pomdp_details.human_collision_penalty,pomdp_details.min_safe_distance_from_obstacle,pomdp_details.obstacle_collision_penalty,
                pomdp_details.radius_around_vehicle_goal,pomdp_details.goal_reached_reward,pomdp_details.max_vehicle_speed,pomdp_details.one_time_step,
                pomdp_details.num_segments_in_one_time_step,pomdp_details.observation_discretization_length,pomdp_details.d_near,
                pomdp_details.d_far,env)

    pomdp_solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound(extended_space_pomdp, b)),max_depth=pomdp_details.tree_search_max_depth),
                        calculate_upper_bound,check_terminal=true),K=pomdp_details.num_scenarios,D=pomdp_details.tree_search_max_depth,T_max=pomdp_details.planning_time,tree_in_info=true)
    pomdp_planner = POMDPs.solve(pomdp_solver, extended_space_pomdp);



    while end_run == false
        println("\n--- --- ---")
        println("k = ", plan_step)
        println("t_k = ", Dates.now())

        # 1: publish current action to ESC
        a_ros = [0.0, 0.0]
        ack_publisher_client(a_ros)

        # 2.a: receive current state and belief
        obs_k = state_updater_client(true)
        belief_ros = belief_updater_client(true)     # TO-DO: need to convert belief_array to actual belief object
        belief_dist_k = ros2pomdp_belief(belief_ros)

        # 3: update environment
        veh_obj = Vehicle(obs_k.state[1],obs_k.state[2],obs_k.state[3],a_ros[1])
        ped_states_k = Array{human_state,1}()
        ped_params_k = Array{human_parameters,1}()
        ped_ids = Array{Int64,1}()
        ped_id = 1
        for i in 4:2:length(obs_k.state)
            ped = human_state(obs_k.state[i], obs_k.state[i+1], 1.0, env_k.goals[1])
            push!(ped_states_k, ped)
            push!(ped_params_k, human_parameters(ped_id))
            push!(ped_ids, ped_id)
            ped_id += 1
        end

        new_sensor_data = vehicle_sensor(ped_states_k, ped_ids, belief_dist_k)
        new_sim_obj = simulator(env,veh_obj,veh_params,new_sensor_data,ped_states_k,ped_params_k,current_sim_obj.one_time_step)

        dist_to_goal = sqrt((veh_obj.x - veh_params.goal.x)^2 + (veh_obj.y - veh_params.goal.y)^2)
        if ((plan_step >= max_plan_steps) || (dist_to_goal <= 1.0))
            end_run = true
            continue
        end

        # 4: calculate action for next cycle with POMDP solver
        future_pred_time = 0.4
        predicted_vehicle_state = propogate_vehicle(new_sim_obj.vehicle, new_sim_obj.vehicle_params,a_ros[2], a_ros[1], future_pred_time)
        modified_vehicle_params = modify_vehicle_params(new_sim_obj.vehicle_params)
        # nearby_humans = get_nearby_humans(new_sim_obj,pomdp_details.num_nearby_humans,pomdp_details.min_safe_distance_from_human,pomdp_details.cone_half_angle)
        b = tree_search_scenario_parameters(predicted_vehicle_state.x,predicted_vehicle_state.y,predicted_vehicle_state.theta,predicted_vehicle_state.v,
                            modified_vehicle_params, exp_details.human_goal_locations, ped_states_k, belief_dist_k, env.length, env.breadth, future_pred_time)
        a_pomdp, info = action_info(pomdp_planner, b)
        a_ros = pomdp2ros_action(a_pomdp, veh_obj.v, planning_Dt, veh_params.L)

        println("vehicle state @ t_k1: ", [predicted_vehicle_state.x, predicted_vehicle_state.y,predicted_vehicle_state.theta])
        println("POMDP action @ t_k1: ", a_pomdp)
        println("ROS action @ t_k1: ", a_ros)

        # 4: book-keeping
        push!(obs_hist, obs_k.state)
        push!(belief_hist, belief_k)
        push!(action_hist, a_ros)
        plan_step += 1

        #5: sleep for remainder of planning loop
        sleep(planning_rate)
    end

    # send [0,0] action to stop vehicle
    ack_publisher_client([0.0, 0.0])
    state_updater_client(false)

    # save state and action history
    @save "/home/adcl/catkin_ws/src/marmot-ros/controller_pkg/histories/o_hist.bson" o_hist
    @save "/home/adcl/catkin_ws/src/marmot-ros/controller_pkg/histories/a_hist.bson" a_hist
end

main()

#=
Changes:
1) Delta Theta Angle to Steering Angle
2) Delta velocity to actual velocity
3) Change [0.0,0.0] to sudden brake action
4) Include sudden brake action
=#


#  # receive initial observation from Vicon, convert to POMDP objects
#  initial_observation = state_updater_client(true)
#  env_right_now.cart.x = initial_observation[1]
#  env_right_now.cart.y = initial_observation[2]
#  env_right_now.cart.theta = initial_observation[3]

#  current_pedestrian_states = Array{human_state,1}()
#  pedestrian_id = 1.0
#  for i in 4:2:length(initial_observation)
#      pedestrian = human_state(initial_observation[i], initial_observation[i+1], 1.0, env_right_now.goals[1], pedestrian_id)
#      pedestrian_id += 1
#      push!(current_pedestrian_states, pedestrian)
#  end

#  env_right_now.complete_cart_lidar_data = current_pedestrian_states
#  env_right_now.cart_lidar_data = current_pedestrian_states

#  # calculate initial belief based on observation
#  initial_belief_over_complete_cart_lidar_data = update_belief([], env_right_now.goals, [], env_right_now.complete_cart_lidar_data)
#  initial_belief = get_belief_for_selected_humans_from_belief_over_complete_lidar_data(initial_belief_over_complete_cart_lidar_data,
#                                                      env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)


# belief_update_time_step = 0.5

#  #Get the first action
#  b = POMDP_2D_action_space_state_distribution(env_right_now,current_belief)
#  action, info = action_info(planner, b)
#  a = pomdp2ros_action(action, env_right_now.cart.v, planning_Dt, env_right_now.cart.L)

# # t=0.0 -> t=0.5 sec
# sleep(belief_update_time_step)
# new_observation = state_updater_client(true)
# new_pedestrian_states = Array{human_state,1}()
# pedestrian_id = 1.0
# for i in 4:2:length(initial_observation)
#     new_pedestrian = human_state(new_observation[i],new_observation[i+1],1.0,env_right_now.goals[1],pedestrian_id)
#     pedestrian_id += 1
#     push!(new_pedestrian_states,new_pedestrian)
# end

# env_right_now.complete_cart_lidar_data = new_pedestrian_states
# env_right_now.cart_lidar_data = new_pedestrian_states
# current_belief_over_complete_cart_lidar_data = update_belief(initial_belief_over_complete_cart_lidar_data, env_right_now.goals, current_pedestrian_states, new_pedestrian_states)
# current_belief = get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
#                                                     env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)
# current_pedestrian_states = new_pedestrian_states

# # t=0.5 -> t=1.0 sec
# sleep(belief_update_time_step)
# new_observation = state_updater_client(true)
# new_pedestrian_states = Array{human_state,1}()
# pedestrian_id = 1.0
# for i in 4:2:length(initial_observation)
#     new_pedestrian = human_state(new_observation[i],new_observation[i+1],1.0,env_right_now.goals[1],pedestrian_id)
#     pedestrian_id += 1
#     push!(new_pedestrian_states,new_pedestrian)
# end

# env_right_now.complete_cart_lidar_data = new_pedestrian_states
# env_right_now.cart_lidar_data = new_pedestrian_states
# current_belief_over_complete_cart_lidar_data = update_belief(current_belief_over_complete_cart_lidar_data, env_right_now.goals,
#                                                         current_pedestrian_states, new_pedestrian_states)
# current_belief = get_belief_for_selected_humans_from_belief_over_complete_lidar_data(current_belief_over_complete_cart_lidar_data,
#                                                     env_right_now.complete_cart_lidar_data, env_right_now.cart_lidar_data)
# current_pedestrian_states = new_pedestrian_states
