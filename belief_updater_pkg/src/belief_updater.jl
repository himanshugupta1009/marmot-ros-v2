#!/usr/bin/env julia

using RobotOS

HAN_path = "/home/adcl/Documents/human_aware_navigation/src/"
include(HAN_path * "struct_definition.jl")
include(HAN_path * "environment.jl")
include(HAN_path * "utils.jl")
include(HAN_path * "pomdp_planning.jl")
include(HAN_path * "belief_tracker.jl")
include(HAN_path * "simulator.jl")
include(HAN_path * "aspen_inputs.jl")

# File Overview:
# - requests state from state_updater, uses to update and store current belief internally (10 Hz)
#   - occurs on its own using a main() function
# - returns belief to controller when requested (2 Hz)

# ROS connections:
#   - to state_updater node as CLIENT to state_updater service
#   - to controller node as PROVIDER of belief_updater service

@rosimport state_updater_pkg.srv: UpdateState
@rosimport belief_updater_pkg.srv: UpdateBelief

rostypegen()
using .state_updater_pkg.srv
using .belief_updater_pkg.srv

# initialize current belief as a global
belief_k = []

# call state_updater service as client
function state_updater_client(record)
    wait_for_service("/car/state_updater/get_state_update")
    update_state_srv = ServiceProxy{UpdateState}("/car/state_updater/get_state_update")

    resp = update_state_srv(UpdateStateRequest(record))

    return resp.state
end

# function called by service
function return_belief(req)
    global belief_k

    # convert belief object to array for ROS message
    belief_array = zeros(Float64, (16,1))
    i = 0

    for human_prob in belief_k
        for prob in human_prob.pdf
            belief_array[i] = prob
            i += 1
        end
    end

    return UpdateBeliefResponse(belief_array)
end

# main function executed by the belief_updater node
function main()
    global belief_k

    # initializes ROS belief_updater node
    init_node("belief_updater")

    # establish belief_updater service as provider
    update_belief_srv = Service{UpdateBelief}(
        "/car/belief_updater/get_belief_update",
        return_belief)

    println("\n--- hello from belief_updater.jl ---\n")
    return 
    
    # sets rate to pull observations and update belief
    obs_Dt = 0.1
    obs_rate = Rate(1/obs_Dt)

    # main loop to repeatedly:
    #   - (x) query Vicon observations
    #   - (o) run belief update
    #   - (x) store current belief in global for controller to request

    # NOTE: don't want belief as a POMDP object, just want 1-d array of belief distributitions
    #   - belief functions may output belief as a POMDP object, need to convert back

    env = generate_environment(5.518, 11.036, obstacle_location[])
    list_human_goals = get_human_goals(env)
    veh_sensor_data = vehicle_sensor(human_state[],Int64[],belief_over_human_goals[])

    while true
        # pull latest observation from state_updater
        obs_k = state_updater_client(true)

        # parse observation into pedestrian states
        peds_k = Array{human_state,1}()
        peds_id = Array{Int64,1}()
        ped_id = 1
        for i in 4:2:length(obs)
            ped = human_state(obs_k[i], obs_k[i+1], 1.0, env.goals[1])
            push!(peds_k, ped)
            push!(peds_id, ped_id)
            ped_id += 1
        end

        # update belief based on observation
        # belief_k_over_complete_cart_lidar_data = update_belief(belief_kn1_over_complete_cart_lidar_data, env.goals, peds_kn1, peds_k)
        # belief_k = get_belief_for_selected_humans_from_belief_over_complete_lidar_data(belief_k_over_complete_cart_lidar_data, ped_states, ped_states)
        belief_k = get_belief(veh_sensor_data, peds_k, peds_id, list_human_goals)
        # get_belief(old_sensor_data, new_lidar_data, new_ids, human_goal_locations)

        # pass variables to next loop
        veh_sensor_data = vehicle_sensor(peds_k, peds_id, belief_k)
        # peds_kn1 = peds_k
        # belief_kn1_over_complete_cart_lidar_data = belief_k_over_complete_cart_lidar_data

        sleep(obs_rate)
    end
end

main()