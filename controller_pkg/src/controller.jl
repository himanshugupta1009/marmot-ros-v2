#!/usr/bin/env julia

using RobotOS
using BSON: @load, @save
using Dates

println("\n--- hello from controller.jl ---\n")

# algs imports
HAN_path = "/home/adcl/Documents/human_aware_navigation/src/"

using Pkg
Pkg.activate("/home/adcl/Documents/BellmanPDEs.jl/")
using BellmanPDEs
using JLD2

# include(HAN_path * "main.jl")
include(HAN_path * "struct_definition.jl")
include(HAN_path * "environment.jl")
include(HAN_path * "utils.jl")
include(HAN_path * "ES_POMDP_Planner.jl")
include(HAN_path * "belief_tracker.jl")
include(HAN_path * "simulator.jl")
include(HAN_path * "parser.jl")
include(HAN_path * "visualization.jl")
include(HAN_path * "HJB_wrappers.jl")
include(HAN_path * "aspen_inputs.jl")

# ROS imports
@rosimport state_updater_pkg.srv: UpdateState
@rosimport belief_updater_pkg.srv: UpdateBelief
@rosimport controller_pkg.srv: UpdateAction

rostypegen()
using .state_updater_pkg.srv
using .belief_updater_pkg.srv
using .controller_pkg.srv


# ROS services ---
# call state_updater service as client
function state_updater_client(record_vicon_history)
    wait_for_service("/car/state_updater/get_state_update")
    update_state_srv = ServiceProxy{UpdateState}("/car/state_updater/get_state_update")

    resp = update_state_srv(UpdateStateRequest(record_vicon_history))

    return resp.state
end

# call belief_updater service as client
function belief_updater_client()
    # println("called BU client")

    wait_for_service("/car/belief_updater/get_belief_update")
    update_belief_srv = ServiceProxy{UpdateBelief}("/car/belief_updater/get_belief_update")

    # println("controller: set up BU client")

    resp = update_belief_srv(UpdateBeliefRequest(true))

    # println("controller: BU respnse: ", resp)

    return resp.belief
end

# call action_updater service as client
function action_updater_client(a)
    wait_for_service("/car/controller/send_action_update")
    update_action_srv = ServiceProxy{UpdateAction}("/car/controller/send_action_update")

    a_req = UpdateActionRequest(a[1], a[2])

    resp = update_action_srv(a_req)
end

# convert POMDP action to ROS control input
function pomdp2ros_action(a_pomdp, v_kn1)
    # pass steering angle
    phi_k = a_pomdp.steering_angle

    # calculate new velocity from Dv
    Dv = a_pomdp.delta_speed
    v_k = v_kn1 + Dv

    return [phi_k, v_k]
end

# TO-DO: need to make sure this aligns with revamped POMDP code
# convert ROS 1D belief array to POMDP belief distribution object
function ros2pomdp_belief(belief_ros)
    belief_pomdp = Array{HumanGoalsBelief,1}()

    for i in 1:4:length(belief_ros)
        b = HumanGoalsBelief(belief_ros[i:i+3])
        push!(belief_pomdp, b)
    end

    return belief_pomdp
end


function main()
    # initialize ROS controller node
    init_node("controller")

    # retrieve scenario config
    input_config = aspen

    # define experiment details and POMDP planning details
    pomdp_details = POMPDPlanningDetails(input_config)
    exp_details = ExperimentDetails(input_config)
    output = OutputObj()

    # define environment
    env = generate_environment(input_config.env_length,input_config.env_breadth,input_config.obstacles)
    exp_details.env = env
    exp_details.human_goal_locations = get_human_goals(env)

    # define vehicle
    veh = Vehicle(input_config.veh_start_x, input_config.veh_start_y, input_config.veh_start_theta, input_config.veh_start_v)
    veh_sensor_data = VehicleSensor(HumanState[],Int64[],HumanGoalsBelief[])
    veh_goal = Location(input_config.veh_goal_x,input_config.veh_goal_y)
    veh_params = VehicleParametersESPlanner(input_config.veh_L,input_config.veh_max_speed,veh_goal)

    # solve HJB equation for the given environment and vehicle
    Dt = 0.5
    max_solve_steps = 200
    Dval_tol = 0.1
    max_steering_angle = 0.475
    HJB_planning_details = HJBPlanningDetails(Dt, max_solve_steps, Dval_tol, max_steering_angle, veh_params.max_speed)
    policy_path = "/home/himanshu/Documents/Research/human_aware_navigation/src"
    
    # solve_HJB = true
    solve_HJB = false

    if(solve_HJB)
        rollout_guide = HJBPolicy(HJB_planning_details, exp_details, veh_params)
        d = Dict("rollout_guide"=>rollout_guide)
        save("/home/adcl/Documents/human_aware_navigation/src/HJB_rollout_guide.jld2",d)
    else
        d = load("/home/adcl/Documents/human_aware_navigation/src/HJB_rollout_guide.jld2")
        rollout_guide = d["rollout_guide"]
    end
    
    # create human and sim objects
    env_humans, env_humans_params = HumanState[], HumanParameters[]
    sim_obj_kn1 = Simulator(env, veh, veh_params, veh_sensor_data, 
        env_humans, env_humans_params, exp_details.simulator_time_step)

    # define POMDP, POMDP solver and POMDP planner
    extended_space_pomdp = ExtendedSpacePOMDP(pomdp_details,exp_details,veh_params,rollout_guide)

    pomdp_solver = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FunctionPolicy(b->calculate_lower_bound(extended_space_pomdp, b)),
        max_depth=pomdp_details.tree_search_max_depth), calculate_upper_bound,check_terminal=true),
        K=pomdp_details.num_scenarios, D=pomdp_details.tree_search_max_depth, T_max=pomdp_details.planning_time, tree_in_info=true)
    
    pomdp_planner = POMDPs.solve(pomdp_solver, extended_space_pomdp);

    # set control loop parameters
    Dt = 0.5
    plan_rate = Rate(1/Dt)

    plan_step = 1
    max_plan_steps = 4 * 60 * 1/Dt
    end_run = false

    a_ros_k = [0.0, 0.0]

    state_hist = []
    action_hist = []


    # TEST ---
    a_ros_k = [0.4, 0.0]
    action_updater_client(a_ros_k)

    obs_k = state_updater_client(true)
    println("obs_k = ", obs_k)

    belief_ros_k = belief_updater_client()
    println("belief_ros_k = ", belief_ros_k)


    # run control loop
    println("controller: starting main loop")
    while end_run == false
        println("\n--- --- ---")
        println("k = ", plan_step, ", t_k = ", Dates.now())

        # 1: publish current action to ESC ---
        # a_ros = [0.0, 0.0]
        action_updater_client(a_ros_k)


        # 2: retrieve current state/observation from Vicon ---
        obs_k = state_updater_client(true)


        # 3: retrieve current belief from belief updater ---
        belief_ros_k = belief_updater_client()
        belief_pomdp_k = ros2pomdp_belief(belief_ros_k)

        # 4: update environment ---
        veh_obj = Vehicle(obs_k[1], obs_k[2], obs_k[3], a_ros_k[2])

        # parse human positions from observation array
        human_states_k = Array{HumanState,1}()
        human_params_k = Array{HumanParameters,1}()
        human_ids = Array{Int64,1}()
        human_id = 1

        for i in 4:2:length(obs_k)
            human = HumanState(obs_k[i], obs_k[i+1], 1.0, exp_details.human_goal_locations[1])

            push!(human_states_k, human)
            push!(human_params_k, HumanParameters(human_id, HumanState[], 1))
            push!(human_ids, human_id)

            human_id += 1
        end

        new_lidar_data, new_ids = human_states_k, human_ids
        sensor_data_k = VehicleSensor(human_states_k, human_ids, belief_pomdp_k)
        sim_obj_k = Simulator(env, veh_obj, veh_params, sensor_data_k, human_states_k, human_params_k, sim_obj_kn1.one_time_step)


        # check terminal conditions
        dist_to_goal = sqrt((veh_obj.x - veh_params.goal.x)^2 + (veh_obj.y - veh_params.goal.y)^2)

        if ((plan_step >= max_plan_steps) || (dist_to_goal <= 1.0))
            end_run = true
            continue
        end


        # 5: calculate action for next cycle with POMDP solver ---
        # propagate vehicle forward to next time step (TO-DO: should be variable based on Date.now() time into loop)
        time_to_k1 = 0.4
        predicted_vehicle_state = propogate_vehicle(sim_obj_k.vehicle, sim_obj_k.vehicle_params, a_ros_k[1], a_ros_k[2], time_to_k1)
        
        # assemble inputs for DESPOT solver
        modified_vehicle_params = modify_vehicle_params(sim_obj_k.vehicle_params)
        b = TreeSearchScenarioParameters(predicted_vehicle_state.x, predicted_vehicle_state.y, predicted_vehicle_state.theta, predicted_vehicle_state.v,
            modified_vehicle_params, exp_details.human_goal_locations, length(human_states_k), human_states_k, belief_pomdp_k,
            sim_obj_k.env.length, sim_obj_k.env.breadth, time_to_k1)
        
        # run DESPOT algorithm
        a_pomdp_k1, info = action_info(pomdp_planner, b)
        a_ros_k1 = pomdp2ros_action(a_pomdp_k1, veh_obj.v)

        println("veh_x_k1 = ", [predicted_vehicle_state.x, predicted_vehicle_state.y, predicted_vehicle_state.theta, predicted_vehicle_state.v])
        println("a_pomdp_k1 = ", a_pomdp_k1, ", a_ros_k1 = ", a_ros_k1)


        # 6: book-keeping ---
        push!(state_hist, obs_k)
        push!(action_hist, a_ros_k)

        plan_step += 1

        a_ros_k = a_ros_k1
        sim_obj_kn1 = sim_obj_k


        # 7: sleep for remainder of planning loop ---
        sleep(plan_rate)
    end

    # send [0,0] action to stop vehicle
    action_updater_client([0.0, 0.0])
    state_updater_client(false)

    # save state and action history
    @save "/home/adcl/catkin_ws/src/marmot-ros/controller_pkg/histories/state_hist.bson" state_hist
    @save "/home/adcl/catkin_ws/src/marmot-ros/controller_pkg/histories/action_hist.bson" action_hist
end

main()


#=
Changes:
1) Change [0.0,0.0] to sudden brake action
2) Include sudden brake action
=#

# env_humans, env_humans_params = generate_humans(env,veh,exp_details.human_start_v,exp_details.human_goal_locations,
#                 exp_details.num_humans_env,exp_details.user_defined_rng,0.1)

# nearby_humans = get_nearby_humans(new_sim_obj, pomdp_details.num_nearby_humans, pomdp_details.min_safe_distance_from_human, pomdp_details.cone_half_angle)

# belief_pomdp_k = get_belief(current_sim_obj.vehicle_sensor_data, new_lidar_data, new_ids, exp_details.human_goal_locations)

# new_sim_obj = Simulator(env, veh_obj, veh_params, sensor_data_k, human_states_k, human_params_k, current_sim_obj.one_time_step)