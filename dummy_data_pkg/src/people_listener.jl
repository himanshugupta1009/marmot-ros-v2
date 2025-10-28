#!/usr/bin/env julia
module PeopleListener
# =============================================================================
# File: people_listener.jl
# Role:
#   Subscribes to /car/dummy/people_poses (PeoplePoseArray), maintains:
#     • latest  — most recent PeoplePoseArray
#     • data_ready (Condition) — signaled once when first human arrives
#     • trajectory_map — per-id history of HumanState for offline use
#     • message_log — (timestamp, msg) tuples for debugging
#   Provides helpers to convert ROS messages → (states, ids, beliefs).
#
# ROS Interface:
#   Subscribes: /car/dummy/people_poses (dummy_data_pkg/PeoplePoseArray)
#
# Key Variables:
#   • latest::Ref{PeoplePoseArray} — single-message cache (overwrite each cb)
#   • data_ready::Condition        — one-shot notify when first human observed
#   • data_notified::Ref{Bool}     — guards one-shot notify
#   • trajectory_map::Dict{Int,Vector{HumanState}} — per-human breadcrumb trail
#   • message_log::Vector{Tuple{Float64,PeoplePoseArray}}> — raw log with times
#
# Assumptions:
#   • Goal assignment for conversions can be round-robin unless provided.
#   • Belief update biases toward goals with positive progress (distance drop).
#
# Gotchas:
#   • DUPLICATE CONST: `data_ready` is declared twice below — keep only one line.
#   • `using ..dummy_data_pkg.msg` assumes this module is nested under a parent
#     where `dummy_data_pkg` is visible. Keep as-is if it works in your tree.
# =============================================================================

    # include(joinpath(@__DIR__, "struct_definition.jl"))
    using RobotOS, Serialization
    @rosimport dummy_data_pkg.msg: PeoplePoseArray
    rostypegen()
    using ..dummy_data_pkg.msg: PeoplePoseArray   # ← note TWO dots (module-relative)

    using Main: HumanState, Location, HumanParameters, HumanGoalsBelief, Vehicle,
                calculate_human_dist_from_all_goals

    # --- live caches & signals ------------------------------------------------
    const latest       = Ref{PeoplePoseArray}(PeoplePoseArray())
    const data_ready   = Condition()  # signaled once when first human appears
    const old_ids      = Ref(Vector{Int}())                 # belief memory
    const old_states   = Ref(Vector{HumanState}())
    const old_beliefs  = Ref(Vector{HumanGoalsBelief}())

    # Offline logs (optional)
    const trajectory_map = Dict{Int, Vector{HumanState}}()
    const message_log    = Vector{Tuple{Float64, PeoplePoseArray}}()

    # NOTE: duplicate declaration (keep only one of these in real code)
    const data_ready = Condition()
    const data_notified = Ref(false)

    """
    _cb(msg::PeoplePoseArray)

    Subscriber callback — updates `latest`, logs message+timestamp, appends
    to `trajectory_map` per id. Notifies `data_ready` once when the first
    human appears.
    """
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
                # Notify once when the first human enters trajectory_map
                if !data_notified[]
                    data_notified[] = true
                    @async notify(data_ready)
                    # println("[PeopleListener] First human added. data_notified = ", data_notified[])
                end
            end
        end
    end

    """
    get_current_humans_and_params(vehicle, goal_locations)
        -> (states, ids, beliefs)

    Converts the latest PeoplePoseArray into planner-friendly arrays,
    computing beliefs based on progress toward each goal.
    """
    function get_current_humans_and_params(vehicle, goal_locations)
        msg = latest[]
        num_goals = length(goal_locations)
        return _rosmsg_to_vectors(msg, num_goals, vehicle, goal_locations)  # (states, ids, beliefs)
    end

    """
    latest_rosmsg_to_humans(msg, goal_locations)
        -> (humans::Vector{HumanState}, params::Vector{HumanParameters})

    Lightweight conversion: assigns goals round-robin; wraps state in
    HumanParameters with a one-element path.
    """
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

    """
    get_human_parameters() -> Vector{HumanParameters}

    Builds HumanParameters from `latest`, using current position as a
    one-point path and a default (0,0) goal placeholder.
    """
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

    # ---- Convert PeoplePoseArray → (states, ids, beliefs) --------------------
    """
    _rosmsg_to_vectors(msg, num_goals, vehicle, goal_locations)
        -> (states, ids, beliefs)

    Belief update heuristic:
      • First sighting of an id → uniform over `num_goals`.
      • Subsequent frames → weight previous belief by per-goal progress
        (distance decrease toward each goal), then renormalize.
    """
    function _rosmsg_to_vectors(msg, num_goals, vehicle, goal_locations)
        n = length(msg.ids)
        states  = Vector{HumanState}(undef, n)
        ids     = Vector{Int64}(undef, n)
        beliefs = Vector{HumanGoalsBelief}(undef, n)

        for i in 1:n
            p = msg.poses[i].position
            h = HumanState(p.x, p.y, 0.0, Location(0, 0))
            states[i] = h
            ids[i] = Int(msg.ids[i])

            # Belief update logic
            idx = findfirst(x -> x == ids[i], old_ids[])
            if isnothing(idx)
                beliefs[i] = HumanGoalsBelief(fill(1/num_goals, num_goals))
            else
                prev_h = old_states[][idx]
                prev_b = old_beliefs[][idx].pdf
                old_dists = calculate_human_dist_from_all_goals(prev_h, goal_locations)
                new_dists = calculate_human_dist_from_all_goals(h, goal_locations)
                progress = old_dists .- new_dists           # positive = moved toward goal
                shift = abs(minimum(progress)) + 1.0        # make strictly positive
                progress .+= shift
                updated_b = prev_b .* progress              # reweight
                beliefs[i] = HumanGoalsBelief(updated_b ./ sum(updated_b))
            end
        end

        # Update memory for next call
        old_ids[]     = copy(ids)
        old_states[]  = copy(states)
        old_beliefs[] = copy(beliefs)

        return states, ids, beliefs
    end

    """
    start_listener()

    Subscribes to /car/dummy/people_poses and spins. Signals `data_ready`
    once when the first human id is observed.
    """
    function start_listener()            # ← blocking version
        # RobotOS.init_node("julia_listener"; anonymous=true)
        RobotOS.Subscriber("/car/dummy/people_poses", PeoplePoseArray, _cb)
        @info "people_listener node running"
        RobotOS.spin()
        # serialize("ped_log.jls", trajectory_map)
        # @info "Wrote $(length(trajectory_map)) pedestrian tracks to ped_log.jls"
    end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PeopleListener.start_listener()
end
