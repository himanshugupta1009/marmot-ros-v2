#!/usr/bin/env julia

using RobotOS

println("\n--- hello from belief_updater.jl ---\n")

# algs imports
HAN_path = "/home/adcl/Documents/human_aware_navigation/src/"

include(HAN_path * "struct_definition.jl")
include(HAN_path * "environment.jl")
include(HAN_path * "utils.jl")
include(HAN_path * "pomdp_planning.jl")
include(HAN_path * "belief_tracker.jl")
include(HAN_path * "simulator.jl")
include(HAN_path * "aspen_inputs.jl")

# ROS imports
@rosimport state_updater_pkg.srv: UpdateState
@rosimport belief_updater_pkg.srv: UpdateBelief

rostypegen()
using .state_updater_pkg.srv
using .belief_updater_pkg.srv


# ROS services ---
# call state_updater service as client
function state_updater_client(record)
    wait_for_service("/car/state_updater/get_state_update")
    update_state_srv = ServiceProxy{UpdateState}("/car/state_updater/get_state_update")

    resp = update_state_srv(UpdateStateRequest(record))

    return resp.state
end

# function run by belief_updater service
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

# (?): which format is the belief global stored as?
# initialize current belief as a global
belief_k = []

function main()
    global belief_k

    # initializes ROS belief_updater node
    init_node("belief_updater")

    # establish belief_updater service as provider
    update_belief_srv = Service{UpdateBelief}(
        "/car/belief_updater/get_belief_update",
        return_belief)

    # set belief loop parameters
    belief_Dt = 0.2
    belief_rate = Rate(1/belief_Dt)


    # REVAMP ---

    env = generate_environment(5.518, 11.036, obstacle_location[])
    list_human_goals = get_human_goals(env)
    veh_sensor_data = vehicle_sensor(human_state[],Int64[],belief_over_human_goals[])

    # ^^^


    while true
        # 1: retrieve current state/observation from Vicon ---
        obs_k = state_updater_client(true)


        # 2: parse human positions from observation array ---
        human_states_k = Array{HumanState,1}()
        human_params_k = Array{HumanParameters,1}()
        human_ids = Array{Int64,1}()
        human_id = 1

        for i in 4:2:length(obs_k.state)
            human = HumanState(obs_k.state[i], obs_k.state[i+1], 1.0, env_k.goals[1])

            push!(human_states_k, human)
            push!(human_params_k, HumanParameters(human_id, HumanState[], 1))
            push!(human_ids, human_id)

            human_id += 1
        end


        # REVAMP ---

        # 3: update belief based on observation ---
        # belief_k_over_complete_cart_lidar_data = update_belief(belief_kn1_over_complete_cart_lidar_data, env.goals, peds_kn1, peds_k)
        # belief_k = get_belief_for_selected_humans_from_belief_over_complete_lidar_data(belief_k_over_complete_cart_lidar_data, ped_states, ped_states)
        belief_k = get_belief(veh_sensor_data, peds_k, peds_id, list_human_goals)
        # get_belief(old_sensor_data, new_lidar_data, new_ids, human_goal_locations)


        # 4: pass variables to next loop ---
        veh_sensor_data = vehicle_sensor(peds_k, peds_id, belief_k)
        # peds_kn1 = peds_k
        # belief_kn1_over_complete_cart_lidar_data = belief_k_over_complete_cart_lidar_data

        # ^^^


        # 5: sleep for remainder of belief loop ---
        sleep(obs_rate)
    end
end

main()


# main loop to repeatedly:
#   - (x) query Vicon observations
#   - (o) run belief update
#   - (x) store current belief in global for controller to request

# NOTE: don't want belief as a POMDP object, just want 1-d array of belief distributitions
#   - belief functions may output belief as a POMDP object, need to convert back