#!/usr/bin/env julia

# algs imports
HAN_env_path = "/home/adcl/Documents/human_aware_navigation/" 
HAN_path = HAN_env_path*"src/"

using Pkg
Pkg.activate(HAN_env_path)


using RobotOS
using BellmanPDEs
using JLD2
import Dates

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


function get_curr_belief!(ros_service_belief, curr_belief, num_goals)
    human_id = 1
    for i in 1:num_goals:length(ros_service_belief)
        pdf = SVector{num_goals,Float64}( view(ros_service_belief, i:i+num_goals-1) )
        b = HumanGoalsBelief(pdf)
        curr_belief[human_id] = b
        human_id += 1
    end
end


struct HardwareExperimentsHelper
    sudden_break_flag::Bool
    run_shield_flag::Bool
    run_with_trials_flag::Bool
    num_humans::Int64
    max_num_planning_steps::Int64
    time_per_step::Float64
    ids::Array{Int64,1}
    human_states::Array{HumanState,1}
    belief::Array{HumanGoalsBelief,1} 
end

function HardwareExperimentsHelper(num_humans)
    ids = collect(1:num_humans)
    human_states = Array{HumanState,1}(undef,num_humans)
    belief = Array{HumanGoalsBelief,1}(undef,num_humans)
    SB_flag = true 
    RS_flag = false
    RWT_flag = false
    δt = 0.5
    max_num_planning_steps = 10
    return HardwareExperimentsHelper(
                            SB_flag,
                            RS_flag,
                            RWT_flag,
                            num_humans,
                            max_num_planning_steps,
                            δt,
                            ids,
                            human_states,
                            belief
                            )
end

struct HardwareExperimentsOutput
    vehicle_state_history::Array{Vehicle,1}
    vehicle_action_history::Array{Tuple{Float64,Float64},1}
end

function HardwareExperimentsOutput(max_steps)
    vehicle_state_history = Array{Vehicle,1}(undef,max_steps)
    vehicle_action_history = Array{Tuple{Float64,Float64},1}(undef,max_steps)
    return HardwareExperimentsOutput(vehicle_state_history,vehicle_action_history)
end

function main(config, code_path, hwe_helper, hwe_output)

    # Initialize ROS controller node
    init_node("controller")

    (;sudden_break_flag, run_shield_flag, run_with_trials_flag, num_humans, max_num_planning_steps, 
                    time_per_step, ids, human_states, belief) =  hwe_helper
    (;vehicle_state_history,vehicle_action_history) = hwe_output
    
    # Define experiment details and POMDP planning details
    pomdp_details = POMPDPlanningDetails(config)
    exp_details = ExperimentDetails(config)
    output = OutputObj()

    # Define Environment
    env = generate_environment(config.env_length,config.env_breadth,config.obstacles)
    exp_details.env = env
    exp_details.human_goal_locations = get_human_goals(env)

    # Define Vehicle
    veh = Vehicle(config.veh_start_x, config.veh_start_y, config.veh_start_theta, config.veh_start_v)
    veh_sensor_data = VehicleSensor(HumanState[],Int64[],HumanGoalsBelief[])
    veh_goal = Location(config.veh_goal_x,config.veh_goal_y)
    r = sqrt( (0.5*config.veh_length)^2 + (0.5*config.veh_breadth)^2 )
    veh_params = VehicleParametersESPlanner(config.veh_wheelbase,config.veh_length,
                    config.veh_breadth,config.veh_dist_origin_to_center, r,
                    config.veh_max_speed,config.veh_max_steering_angle,veh_goal)
    vehicle_body = get_vehicle_body((veh_params.length,veh_params.length), (veh_params.dist_origin_to_center,0.0))
    output.vehicle_body = vehicle_body
    
    # Define Humans
    env_humans, env_humans_params = HumanState[], HumanParameters[]

    # Create sim object
    navigation_sim_obj = NavigationSimulator(env,veh,veh_params,veh_sensor_data,
                        env_humans,env_humans_params,exp_details.simulator_time_step)

    # Access Rollout guide 
    s = load(code_path * "rollout_guides/HJB_rollout_guide_aspen.jld2")
    rollout_guide = s["rollout_guide"]
    
    
    #=
    Define POMDP, POMDP Solver and POMDP Planner
    =#    
    custom_pomdp_planning_time = 0.4  #Use if needed
    sol_rng = MersenneTwister(19)
    SB_flag = sudden_break_flag  #Apply Sudden Break Flag
    extended_space_pomdp = ExtendedSpacePOMDP(pomdp_details,env,veh_params,rollout_guide,SB_flag);
    lower_bound_func = DefaultPolicyLB(
                            FunctionPolicy(b->calculate_lower_bound(extended_space_pomdp, b)),
                            max_depth=pomdp_details.tree_search_max_depth
                            )
    upper_bound_func = old_calculate_upper_bound
    run_with_trials = run_with_trials_flag
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
    RS_flag = run_shield_flag  #Run shield Flag
    shield_utils = ShieldUtils(exp_details, pomdp_details, veh_params.wheelbase)


    sleep(2.0)
    println("Controller: Sending Test Requests to Services")

    # TEST ---
    action_updater_client((0.0, 0.0))
    test_obs = state_updater_client(true)
    println("Test Vicon Observation: ", test_obs)
    test_belief_ros = belief_updater_client()
    println("Test Belief: ", test_belief_ros)


    # Set control loop parameters
    planning_rate = Rate(1/time_per_step)
    planning_step = 1
    curr_vehicle_speed = 0.0
    curr_vehicle_ϕ = 0.0
    curr_vehicle_action = (curr_vehicle_ϕ,curr_vehicle_speed)
    end_experiment = false
    placeholder_human_goal = exp_details.human_goal_locations[1]
    num_human_goals = length(exp_details.human_goal_locations)


    # Run the control loop
    println("Controller: Starting the while loop in the main function")
    while( end_experiment == false  && planning_step < max_num_planning_steps )

        println("Planning Loop #", planning_step, ", Time Stamp = ", Dates.now())

        #=
        1: Retrieve environment state for current planning loop frssom Vicon
        =#
        curr_obs = state_updater_client(true)

        #=
        Define the vehicle state and print it and the current action.
        =#  
        curr_vehicle = Vehicle(curr_obs[1], curr_obs[2], curr_obs[3], curr_vehicle_speed)
        println("Observed Vehicle State at the beginning of Planning Loop #$(planning_step): ", curr_vehicle)
        println("Vehicle Action for Planning Loop #$(planning_step): (ϕ: $curr_vehicle_ϕ; s: $curr_vehicle_speed)")

        #=
        2: Publish the current vehicle action to ESC (the motor controller)
        =#
        # if ( planning_step==2 || planning_step==3 )  # TO-DO: Make this Optional or even remove this if block.
        #     curr_vehicle_action = (0.0, 0.3)
        # end
        curr_vehicle_action = (0.0,0.25)
        action_updater_client(curr_vehicle_action)
        
        #= 
        3: retrieve current belief from belief updater
        =#
        curr_belief_from_ros_service = belief_updater_client()
        get_curr_belief!(curr_belief_from_ros_service,belief,num_human_goals)
        curr_belief = belief 

        #=
        4: Update Human positions and Vehicle's belief for determining next vehicle actions by POMDP tree search
        =#
        curr_human_states = human_states
        human_id = 1
        for i in 5:2:length(curr_obs)   #TO-DO: Modify the structure to get rid of these numbers 
            human = HumanState(curr_obs[i], curr_obs[i+1], 1.0, placeholder_human_goal)
            curr_human_states[human_id] = human
            human_id += 1
        end
        # println("Human States for Planning Loop #$(planning_step) : ", round.(curr_obs[5:end], digits=3))

        #=
        5: Calculate Vehicle Action for the next cycle with POMDP tree search
        (from Will) propagate vehicle forward to next time step (TO-DO: should be variable based on Date.now() time into loop)
        =#
        time_duration_until_pomdp_action_determined = 0.5
        predicted_vehicle_state = propogate_vehicle(curr_vehicle, veh_params, curr_vehicle_ϕ, 
                                    curr_vehicle_speed,time_duration_until_pomdp_action_determined)
        predicted_vehicle_center_x = predicted_vehicle_state.x + veh_params.dist_origin_to_center*cos(predicted_vehicle_state.theta)
        predicted_vehicle_center_y = predicted_vehicle_state.y + veh_params.dist_origin_to_center*sin(predicted_vehicle_state.theta)
        println("Predicted Vehicle position for Planning Loop #$(planning_step+1) = ", predicted_vehicle_state)

        println("Human Position : $curr_human_states")
        println("Belief : $curr_belief")
    
        # Check if the predicted vehicle position is in goal!
        dist_to_goal = sqrt((predicted_vehicle_center_x - veh_params.goal.x)^2 + (predicted_vehicle_center_y - veh_params.goal.y)^2)
        if (dist_to_goal <= exp_details.radius_around_vehicle_goal)
            println("Predicted Vehicle position in Planning Loop #$(planning_step+1) is in goal. Vehicle will reach goal in the next planning step!")
            end_experiment = true
            next_vehicle_action = (-10.0,-10.0)
        else
            # Define the Scenario Parameters for POMDP tree search
            b = TreeSearchScenarioParameters(predicted_vehicle_state.x, predicted_vehicle_state.y, predicted_vehicle_state.theta, predicted_vehicle_state.v,
                            veh_params, exp_details.human_goal_locations, length(curr_human_states), curr_human_states, curr_belief,
                            env.length, env.breadth, time_duration_until_pomdp_action_determined)
        
            # Run the DESPOT algorithm
            next_vehicle_action_pomdp, info = action_info(pomdp_planner, b)
            # next_vehicle_action = (next_vehicle_action_pomdp.steering_angle, next_vehicle_action_pomdp.delta_speed)
            next_vehicle_action = (0.0,0.0)
        end

        println("POMDP action for the next loop: $next_vehicle_action")
        println("*************************************************************************************************")


        #= From Will
            ISSUE: need to be using most recent velocity measurement when computing next velocity command
          - should be taking a new observation just before execution, after DESPOT and sleep(rate)
        =#

        #=
        6: Book Keeping
        =#
        vehicle_state_history[planning_step] = curr_vehicle
        vehicle_action_history[planning_step] = curr_vehicle_action 

        # Format: next_vehicle_action = (ϕ,s)
        curr_vehicle_ϕ = next_vehicle_action[1]
        curr_vehicle_speed = clamp(curr_vehicle_speed+next_vehicle_action[2], 0.0, pomdp_details.max_vehicle_speed)
        curr_vehicle_action = (curr_vehicle_ϕ,curr_vehicle_speed)
        planning_step += 1

        #=
        7: Sleep for the remainder of the planning loop
        =#
        sleep(planning_rate)

    end

    # Send (0.0,0.0) action to stop the vehicle
    action_updater_client((0.0, 0.0)) 
    state_updater_client(false)
    return

    # save state and action history
    # @save "/home/adcl/catkin_ws/src/marmot-ros-v2/controller_pkg/histories/state_hist.bson" state_hist
    # @save "/home/adcl/catkin_ws/src/marmot-ros-v2/controller_pkg/histories/action_hist.bson" action_hist
end

marmot_experiments_helper = HardwareExperimentsHelper(4)
marmot_experiments_output = HardwareExperimentsOutput(50)
main(input_config, HAN_path, marmot_experiments_helper, marmot_experiments_output)


#=
Changes:
1) Change [0.0,0.0] to sudden brake action
2) Include sudden brake action
=#

#= Psuedo Code

while(not_end)
    Get Vehicle and Human position
    Publish Vehicle Action computed at previous time step
    Get the belief over humans
    Predict the Vehicle position at the beginning of the next cycle
    Check if it is in goal
    If not, compute the next vehicle action using DESPOT
    (Optional) Compute Shield and Get the Safe Action

=#