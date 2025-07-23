#!/usr/bin/env julia            
module PeopleListener

    include(joinpath(@__DIR__, "struct_definition.jl"))
    using RobotOS, Serialization
    @rosimport dummy_data_pkg.msg: PeoplePoseArray
    rostypegen()
    using ..dummy_data_pkg.msg: PeoplePoseArray   # ← note TWO dots

    # import ..Main: HumanState, Location, HumanParameters, Vehicle, HumanGoalsBelief
    const latest = Ref{PeoplePoseArray}(PeoplePoseArray())
    const old_ids     = Ref(Vector{Int}())
    const old_states  = Ref(Vector{HumanState}())
    const old_beliefs = Ref(Vector{HumanGoalsBelief}())

    const trajectory_map = Dict{Int, Vector{HumanState}}()
    const message_log = Vector{Tuple{Float64, PeoplePoseArray}}()

    function _cb(msg::PeoplePoseArray)
        latest[] = msg
        timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
        # @info "rx PeoplePoseArray Δt=$(round(timestamp; digits=2))s  humans=$(length(msg.ids))"

        for (i, id32) in enumerate(msg.ids)
            id = Int(id32)
            pos = msg.poses[i].position
            h = HumanState(pos.x, pos.y, 0.0, Location(0.0, 0.0))  # goal may be updated later

            push!(message_log, (timestamp, deepcopy(msg)))
            if haskey(trajectory_map, id)
                push!(trajectory_map[id], h)
            else
                trajectory_map[id] = [h]
            end
        end
    end


    function get_current_humans_and_params(vehicle::Vehicle, goal_locations::Vector{Location})
        msg = latest[]
        num_goals = length(goal_locations)
        return Main._rosmsg_to_vectors(msg, num_goals, vehicle, goal_locations;)  # returns (states, ids, beliefs)
    end

    function latest_rosmsg_to_humans(msg::PeoplePoseArray, goal_locations::Vector{Location})
        n = length(msg.ids)
        humans = Vector{HumanState}(undef, n)
        params = Vector{HumanParameters}(undef, n)

        for i in 1:n
            id = Int(msg.ids[i])
            p = msg.poses[i].position
            goal_id = mod(id, length(goal_locations)) + 1  # round-robin or any logic
            goal = goal_locations[goal_id]

            humans[i] = HumanState(p.x, p.y, 0.0, goal)
            params[i] = HumanParameters(id, [humans[i]], 1)  # path = just current pos
        end
        return humans, params
    end


    function get_human_parameters()
        msg = latest[]
        n = length(msg.ids)
        params = HumanParameters[]

        for i in 1:n
            x = msg.poses[i].position.x
            y = msg.poses[i].position.y
            id = Int(msg.ids[i])
            hs = HumanState(x, y, 0.0, Location(0, 0))
            push!(params, HumanParameters(id, [hs], 1))
        end

        return params
    end


    function start_listener()            # ← blocking version
        RobotOS.init_node("people_listener"; anonymous=true)
        RobotOS.Subscriber("/car/dummy/people_poses", PeoplePoseArray, _cb)
        @info "people_listener node running — press Ctrl-C to stop"
        RobotOS.spin()                   # block until Ctrl-C / rosnode kill
        serialize("ped_log.jls", trajectory_map)
        @info "Wrote $(length(trajectory_map)) pedestrian tracks to ped_log.jls"
    end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PeopleListener.start_listener()
end