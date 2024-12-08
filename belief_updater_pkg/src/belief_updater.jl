#!/usr/bin/env julia

# algs imports
HAV_env_path = "/home/adcl/Documents/human_aware_navigation/" 
HAN_path = HAV_env_path*"src/"

using Pkg
Pkg.activate(HAV_env_path)
using RobotOS

println("\n--- hello from belief_updater.jl ---\n")

include(HAN_path * "struct_definition.jl")
include(HAN_path * "parser.jl")
include(HAN_path * "environment.jl")
include(HAN_path * "utils.jl")
include(HAN_path * "belief_tracker.jl")
include(HAN_path * "simulator.jl")
include(HAN_path * "simulator_utils.jl")
include(HAN_path * "configs/aspen.jl")

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

    # println("BU: received CT request")

    # convert belief object to array for ROS message
    belief_array = zeros(Float64, 16)
    i = 1
    # println("BU: belief_array = ", belief_array)
    for human_prob in belief_k
        # println("human_prob , " , human_prob.pdf)
        for prob in human_prob.pdf
            # println("PP: ,", prob)
            belief_array[i] = prob
            i += 1
        end
    end

    # println(belief_array)
    # println("Before returning")
    # println(typeof(belief_array))
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

    # POMDP params
    pomdp_details = POMPDPlanningDetails(input_config)
    exp_details = ExperimentDetails(input_config)
    env = generate_environment(input_config.env_length,input_config.env_breadth,input_config.obstacles)
    exp_details.env = env
    exp_details.human_goal_locations = get_human_goals(env)
    veh_sensor_data_kn1 = VehicleSensor(HumanState[],Int64[],HumanGoalsBelief[])

    # set belief loop parameters
    belief_Dt = 0.2
    belief_rate = Rate(1/belief_Dt)

    while true
        # 1: retrieve current state/observation from Vicon ---
        obs_k = state_updater_client(true)


        # 2: parse human positions from observation array ---
        human_states_k = Array{HumanState,1}()
        human_params_k = Array{HumanParameters,1}()
        human_ids = Array{Int64,1}()
        human_id = 1

        for i in 5:2:length(obs_k)
            human = HumanState(obs_k[i], obs_k[i+1], 1.0, exp_details.human_goal_locations[1])

            push!(human_states_k, human)
            push!(human_params_k, HumanParameters(human_id, HumanState[], 1))
            push!(human_ids, human_id)

            human_id += 1
        end


        # 3: update belief based on observation ---
        new_lidar_data, new_ids = human_states_k, human_ids
        belief_k = get_belief(veh_sensor_data_kn1, new_lidar_data, 
                                        new_ids, exp_details.human_goal_locations)


        # println("belief_updater: belief_k = ", belief_k)


        # 4: pass variables to next loop ---
        veh_sensor_data_k = VehicleSensor(human_states_k, human_ids, belief_k)
        veh_sensor_data_kn1 = veh_sensor_data_k


        # 5: sleep for remainder of belief loop ---
        sleep(belief_rate)
    end
end

main()