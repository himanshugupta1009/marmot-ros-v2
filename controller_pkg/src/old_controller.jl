#!/usr/bin/env julia

# algs imports
HAV_env_path = "/home/adcl/Documents/human_aware_navigation/" 
HAN_path = HAV_env_path*"src/"

using Pkg
Pkg.activate(HAV_env_path)


using RobotOS
using BellmanPDEs
using JLD2
using Dates

println("\n--- hello from controller.jl ---\n")

include(HAN_path * "struct_definition.jl")
include(HAN_path * "environment.jl")
include(HAN_path * "utils.jl")
include(HAN_path * "ES_POMDP_Planner.jl")
include(HAN_path * "belief_tracker.jl")
include(HAN_path * "simulator.jl")
include(HAN_path * "simulator_utils.jl")
include(HAN_path * "parser.jl")
include(HAN_path * "visualization.jl")
include(HAN_path * "HJB_wrappers.jl")
include(HAN_path * "shielding/shield_utils.jl")
include(HAN_path * "shielding/shield.jl")

#Include the Config File for ASPEN
include(HAN_path * "configs/aspen.jl")



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

    # println("controller: set up bu client")

    resp = update_belief_srv(UpdateBeliefRequest(true))

    # println("controller: bu response: ", resp)

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
function pomdp2ros_action(a_pomdp, v_kn1, max_v_speed)
    # pass steering angle
    phi_k = a_pomdp.steering_angle

    # calculate new velocity from Dv
    Dv = a_pomdp.delta_speed
    if(Dv==-10.0)
        return [0.0, 0.0]
    else
        v_k = clamp(v_kn1 + Dv, 0.0, max_v_speed)
        return [phi_k, v_k]
    end

end

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
    aspen = input_config

    # define experiment details and POMDP planning details
    pomdp_details = POMPDPlanningDetails(input_config)
    exp_details = ExperimentDetails(input_config)
    output = OutputObj()

    #Define Environment
    env = generate_environment(input_config.env_length,input_config.env_breadth,input_config.obstacles)
    exp_details.env = env
    exp_details.human_goal_locations = get_human_goals(env)

    #Define Vehicle
    veh = Vehicle(input_config.veh_start_x, input_config.veh_start_y, input_config.veh_start_theta, input_config.veh_start_v)
    veh_sensor_data = VehicleSensor(HumanState[],Int64[],HumanGoalsBelief[])
    veh_goal = Location(input_config.veh_goal_x,input_config.veh_goal_y)
    r = sqrt( (0.5*input_config.veh_length)^2 + (0.5*input_config.veh_breadth)^2 )
    veh_params = VehicleParametersESPlanner(input_config.veh_wheelbase,input_config.veh_length,
                    input_config.veh_breadth,input_config.veh_dist_origin_to_center, r,
                    input_config.veh_max_speed,input_config.veh_max_steering_angle,veh_goal)
    vehicle_body = get_vehicle_body((veh_params.length,veh_params.length), (veh_params.dist_origin_to_center,0.0))
    output.vehicle_body = vehicle_body
    
    #=
    Define Humans
    =#
    env_humans, env_humans_params = HumanState[], HumanParameters[]

    #=
    Create sim object
    =#
    sim_obj_kn1 = NavigationSimulator(env,veh,veh_params,veh_sensor_data,
                        env_humans,env_humans_params,exp_details.simulator_time_step)


    # Access Rollout guide 
    s = load(HAN_path * "rollout_guides/HJB_rollout_guide_aspen.jld2")
    rollout_guide = s["rollout_guide"]
    
    
    #=
    Define POMDP, POMDP Solver and POMDP Planner
    =#    
    custom_pomdp_planning_time = 0.4  #use if needed
    sol_rng = MersenneTwister(19)
    SB_flag = false  #Apply Sudden Break Flag
    SB_flag = true  #Apply Sudden Break Flag
    extended_space_pomdp = ExtendedSpacePOMDP(pomdp_details,env,veh_params,rollout_guide,SB_flag);
    lower_bound_func = DefaultPolicyLB(
                            FunctionPolicy(b->calculate_lower_bound(extended_space_pomdp, b)),
                            max_depth=pomdp_details.tree_search_max_depth
                            )
    upper_bound_func = old_calculate_upper_bound
    run_with_trials = false
    # run_with_trials = true
    pomdp_solver = DESPOTSolver(
                        bounds=IndependentBounds(lower_bound_func,upper_bound_func,check_terminal=true,consistency_fix_thresh=1e-5),
                        K=pomdp_details.num_scenarios,D=pomdp_details.tree_search_max_depth,
                        tree_in_info=true,
                        T_max = run_with_trials ? Inf : pomdp_details.planning_time,
                        max_trials = run_with_trials ? pomdp_details.max_num_trials : 100,
                        default_action=get_default_action,
                        rng=sol_rng
                        )
    pomdp_planner = POMDPs.solve(pomdp_solver, extended_space_pomdp);
    
    #=
    Define Shield Utils
    =#
    RS_flag = true  #Run shield Flag
    RS_flag = false  #Run shield Flag
    shield_utils = ShieldUtils(exp_details, pomdp_details, veh_params.wheelbase)


    # set control loop parameters
    Dt = 0.5
    plan_rate = Rate(1/Dt)

    plan_step = 1
    max_plan_steps = 4 * 60 * 1/Dt
    end_run = false

    a_pomdp_k = ActionExtendedSpacePOMDP(0.0, 0.0)

    state_hist = []
    action_hist = []

    sleep(2.0)
    println("controller: sending test requests to services")

    # TEST ---
    action_updater_client([0.4, 0.0])

    obs_k = state_updater_client(true)
    println("obs_k = ", obs_k)

    belief_ros_k = belief_updater_client()
    println("belief_ros_k = ", belief_ros_k)

    # println("controller: going into sleep loop") 
    # while true
    #     sleep(0.5)
    # end
    # ---

    # run control loop
    println("controller: starting main loop")
    while end_run == false
        println("\n--- --- ---")
        println("k = ", plan_step, ", t_k = ", Dates.now())

        # 1: retrieve state at t_k- from Vicon ---
        obs_k = state_updater_client(true)

        x_km = obs_k[1:4]
        println("x_km observed = ", x_km)

        # 2: publish current action to ESC ---
        v_km = clamp(x_km[4], 0.0, 2.0)    # true velocity just before t_k

        a_ros_k = pomdp2ros_action(a_pomdp_k, v_km, pomdp_details.max_vehicle_speed)    # add requested Dv to true v_km

        # NOTE: make sure this works correctly with long first step
        if plan_step == 1
            a_ros_k = [0.0, 0.0]
        elseif plan_step in [2, 3]
            a_ros_k = [0.0, 0.3]
        end

        action_updater_client(a_ros_k)

        v_kp = a_ros_k[2]   # "perfect step" velocity just after t_k

        println("a_pomdp_k = $a_pomdp_k, v_km = $v_km, a_ros_k = $a_ros_k, v_kp = $v_kp")


        # 3: retrieve current belief from belief updater ---
        belief_ros_k = belief_updater_client()
        belief_pomdp_k = ros2pomdp_belief(belief_ros_k)


        # 4: update environment ---
        # define vehicle object at t_k+
        veh_obj = Vehicle(obs_k[1], obs_k[2], obs_k[3], v_kp)   # other states should be unchanged from t_k- to t_k+
        println("x_kp used for POMDP = ", veh_obj)

        # parse human positions from observation array
        human_states_k = Array{HumanState,1}()
        human_params_k = Array{HumanParameters,1}()
        human_ids = Array{Int64,1}()
        human_id = 1

        println("humans_k = ", round.(obs_k[5:end], digits=3))

        for i in 5:2:length(obs_k)
            human = HumanState(obs_k[i], obs_k[i+1], 1.0, exp_details.human_goal_locations[1])

            push!(human_states_k, human)
            push!(human_params_k, HumanParameters(human_id, HumanState[], 1))
            push!(human_ids, human_id)

            human_id += 1
        end

        new_lidar_data, new_ids = human_states_k, human_ids
        sensor_data_k = VehicleSensor(human_states_k, human_ids, belief_pomdp_k)
        sim_obj_k = Simulator(env, veh_obj, veh_params, sensor_data_k, human_states_k, human_params_k, sim_obj_kn1.one_time_step)


        # 5: calculate action for next cycle with POMDP solver ---
        # propagate vehicle forward to next time step (TO-DO: should be variable based on Date.now() time into loop)
        time_to_k1 = 0.5
        predicted_vehicle_state = propogate_vehicle(sim_obj_k.vehicle, sim_obj_k.vehicle_params, a_ros_k[1], a_ros_k[2], time_to_k1)
        
        # check terminal conditions
        dist_to_goal = sqrt((predicted_vehicle_state.x - veh_params.goal.x)^2 + (predicted_vehicle_state.y - veh_params.goal.y)^2)
        if ((plan_step >= max_plan_steps) || (dist_to_goal <= exp_details.radius_around_vehicle_goal))
            end_run = true
            continue
        end

        # assemble inputs for DESPOT solver
        modified_vehicle_params = modify_vehicle_params(sim_obj_k.vehicle_params)
        b = TreeSearchScenarioParameters(predicted_vehicle_state.x, predicted_vehicle_state.y, predicted_vehicle_state.theta, predicted_vehicle_state.v,
            modified_vehicle_params, exp_details.human_goal_locations, length(human_states_k), human_states_k, belief_pomdp_k,
            sim_obj_k.env.length, sim_obj_k.env.breadth, time_to_k1)
        
        # run DESPOT algorithm
        a_pomdp_k1, info = action_info(pomdp_planner, b)

        # ISSUE: need to be using most recent velocity measurement when computing next velocity command
        #   - should be taking a new observation just before execution, after DESPOT and sleep(rate)

        println("veh_x_k1 predicted = ", [predicted_vehicle_state.x, predicted_vehicle_state.y, predicted_vehicle_state.theta, predicted_vehicle_state.v])


        # 6: book-keeping ---
        push!(state_hist, obs_k)
        push!(action_hist, a_ros_k)

        a_pomdp_k = a_pomdp_k1
        sim_obj_kn1 = sim_obj_k

        plan_step += 1


        # 7: sleep for remainder of planning loop ---
        sleep(plan_rate)
    end

    # send [0,0] action to stop vehicle
    action_updater_client([0.0, 0.0])
    state_updater_client(false)

    # save state and action history
    @save "/home/adcl/catkin_ws/src/marmot-ros-v2/controller_pkg/histories/state_hist.bson" state_hist
    @save "/home/adcl/catkin_ws/src/marmot-ros-v2/controller_pkg/histories/action_hist.bson" action_hist
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