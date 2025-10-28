#!/usr/bin/env julia
# =============================================================================
# File: realtime_vehicle_node.jl
# Role:
#   Realtime vehicle loop that:
#     • receives scheduled actions via /set_action,
#     • integrates vehicle state in 0.1 s inner steps over a 0.5 s block,
#     • fuses pedestrian data (PeopleListener) with LiDAR prefiltering,
#     • computes nearby-human beliefs,
#     • publishes a compact LiveData snapshot on /car/sim/LiveData,
#     • (optionally) builds a GIF from runtime logs when finished.
#
# ROS Interfaces
#   Service (server): /set_action (dummy_data_pkg/SetAction)
#       req:  t, steering, speed  → stores into ACTION_ARRAY at 0.5 s grid key
#   Topic (pub)     : /car/sim/LiveData (dummy_data_pkg/LiveData)
#       pose/vel + filtered humans and beliefs for the current inner step
#   Topic (sub, ext): /car/dummy/people_poses (via PeopleListener)
#
# Timing
#   • INNER step  DT = 0.1 s
#   • OUTER block TOTAL_TIME = 0.5 s (5 inner steps)
#   • Loop prints timing breakdown per inner step (move, people, lidar, sim, nearby, publish)
#
# Key Variables
#   • ENV, VEHICLE_PARAMS, EXP_DETAILS    — from main_es.jl (world, vehicle, timing)
#   • GOALS                               — EXP_DETAILS.human_goal_locations
#   • ACTION_ARRAY::Dict{Float64,(Float64,Float64)}
#       maps block key time (seconds, .0 or .5) → (steering [rad], speed [m/s])
#   • VEHICLE_TRAJ::Dict{Float64,Vehicle}     (provided externally; this file writes)
#   • BELIEF_ARRAY::Dict{Float64,Tuple{Vector{HumanState},Vector{Int64},Vector{HumanGoalsBelief}}}
#       (provided externally; this file writes)
#   • PeopleListener.*                    — listener that provides latest pedestrian data & a Condition
#
# Assumptions
#   • PeopleListener is initialized and will signal `data_ready` within 20 s (configurable in `main()`).
#   • VEHICLE_TRAJ and BELIEF_ARRAY exist in the current Julia process (often from main_es.jl include).
#
# Gotchas
#   • The loop code uses `veh` in a few places but the current vehicle is `curr_veh`.
#     That’s likely a typo; behavior kept as-is here for non-intrusive commenting.
#   • LIDAR_RANGE is hard-set to 200 (overrides EXP_DETAILS.lidar_range).
#   • `ACTION_ARRAY` key uses `round(req.t/0.5)*0.5` (nearest 0.5 s); client should schedule on the 0.5 s grid.
# =============================================================================

import Pkg; Pkg.activate(joinpath(@__DIR__))
using RobotOS
@rosimport dummy_data_pkg.msg: PeoplePoseArray, LiveData, BeliefArray
@rosimport dummy_data_pkg.srv: GetAction, GetLiveData, SetAction
rostypegen()
const PoseMsg = geometry_msgs.msg.Pose
using .dummy_data_pkg.msg: PeoplePoseArray, LiveData, BeliefArray
using .dummy_data_pkg.srv: GetLiveData, GetLiveDataRequest, GetLiveDataResponse, GetAction, GetActionRequest, GetActionResponse, SetAction, SetActionRequest, SetActionResponse

# === Include true environment context ===
include(joinpath(@__DIR__, "struct_definition.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src", "main_es.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","simulator_utils.jl"))
include(joinpath(@__DIR__, "people_listener.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src", "belief_tracker.jl"))

using JSON
using DataStructures: OrderedDict
using Plots, LazySets, StaticArrays
include(joinpath(@__DIR__, "..", "..", "..","..","src", "human_aware_navigation_modifiedbyansh", "src", "visualization.jl"))

# Pull simulation constants from main_es.jl
const ENV = Main.env
const VEHICLE_PARAMS = Main.veh_params
const EXP_DETAILS = Main.exp_details
const GOALS = EXP_DETAILS.human_goal_locations

# Constants for LiDAR-based filtering
const DT = EXP_DETAILS.simulator_time_step        # 0.1
const TOTAL_TIME = 0.5
const N_STEPS = Int(TOTAL_TIME / DT)

const LIDAR_RANGE = 200 # EXP_DETAILS.lidar_range   # NOTE: hard override
const NUM_NEARBY_HUMANS = Main.pomdp_details.num_nearby_humans
const MIN_SAFE_DIST = EXP_DETAILS.min_safe_distance_from_human
const CONE_HALF_ANGLE = Main.pomdp_details.cone_half_angle

# Action store used by /set_action service
const ACTION_ARRAY = Dict{Float64, Tuple{Float64, Float64}}()

const MAX_TIME_LIMIT = 50.0

@rosimport dummy_data_pkg.msg: LiveData
rostypegen()
using .dummy_data_pkg.msg: LiveData

"""
publish_live_data(veh, nbh)

Builds a LiveData message from the current vehicle `veh` and the computed
NearbyHumans `nbh`, publishes it on /car/sim/LiveData, and updates LATEST.

Fields included:
  x, y, theta, v (Float32), ids (Int32[]), hx/hy (Float32[]), belief_pdf (Float32[])
"""
function publish_live_data(veh, nbh)
    msg = LiveData()
    msg.header.stamp     = RobotOS.now()
    msg.header.frame_id  = "world"

    msg.x      = Float32(veh.x)
    msg.y      = Float32(veh.y)
    msg.theta  = Float32(veh.theta)
    msg.v      = Float32(veh.v)

    N          = length(nbh.ids)
    num_goals  = length(EXP_DETAILS.human_goal_locations)

    resize!(msg.ids, N)
    resize!(msg.hx,  N)
    resize!(msg.hy,  N)
    resize!(msg.belief_pdf, N*num_goals)

    for i in 1:N
        msg.ids[i] = Int32(nbh.ids[i])
        msg.hx[i]  = Float32(nbh.position_data[i].x)
        msg.hy[i]  = Float32(nbh.position_data[i].y)
        b = nbh.belief[i].pdf
        for g in 1:num_goals
            msg.belief_pdf[(i-1)*num_goals + g] = Float32(b[g])
        end
    end
    LATEST[] = msg
    RobotOS.publish(LIVE_PUB, msg)
end

"""
handle_set_action(req::SetActionRequest) -> SetActionResponse

Quantizes the requested time to the nearest 0.5 s grid and stores the
(steering, speed) pair into ACTION_ARRAY keyed by that time.
"""
function handle_set_action(req::SetActionRequest)::SetActionResponse
    t = Int(round(req.t / 0.5))*0.5   # nearest 0.5-s block
    ACTION_ARRAY[t] = (req.steering, req.speed)
    msg = "Stored action at t=$t"
    @info msg
    return SetActionResponse(true, "")
end

"""
warmup_realtime_functions!(; n_humans=30, repeats=3)

JIT-warm core runtime paths: vehicle propagation, LiDAR prefilter, simulator
wrapper, nearby-human computation, and writing VEHICLE_TRAJ/BELIEF_ARRAY.
"""
function warmup_realtime_functions!(; n_humans=30, repeats=3)
    # 1) vehicle in the middle of the map
    veh = Vehicle(ENV.length/2, ENV.breadth/2, 0.0, 1.0)

    # 2) People around the vehicle with valid IDs/goals
    humans = HumanState[]
    hparams = HumanParameters[]
    for k in 1:n_humans
        θ = 2π*k/n_humans
        x = veh.x + 2.0*cos(θ)
        y = veh.y + 2.0*sin(θ)
        goal = GOALS[mod1(k, length(GOALS))]
        push!(humans, HumanState(x, y, 1.0, goal))
        push!(hparams, HumanParameters(k, [humans[end]], 1))
    end
    ids = [hp.id for hp in hparams]
    beliefs = [default_human_goal_belief(length(GOALS)) for _ in ids]

    # 3) One sensor snapshot (LiDAR prefilter uses the same code as your loop)
    lidar_data, lidar_ids = get_lidar_data_and_ids(veh, humans, hparams, LIDAR_RANGE)
    sensor = VehicleSensor(lidar_data, lidar_ids, beliefs[1:length(lidar_ids)])

    # 4) Build the simulator wrapper (same type as in the loop)
    sim = NavigationSimulator(ENV, veh, VEHICLE_PARAMS, sensor, humans, hparams, DT)

    # 5) Walk through the exact inner-loop path a few times to JIT everything
    for r in 1:repeats
        # propagate vehicle
        nx, ny, nθ = move_vehicle(veh.x, veh.y, veh.theta, VEHICLE_PARAMS.wheelbase, 0.02, 1.0, DT)
        veh = Vehicle(nx, ny, nθ, 1.0)

        # LiDAR snapshot again at the new pose
        lidar_data, lidar_ids = get_lidar_data_and_ids(veh, humans, hparams, LIDAR_RANGE)
        sensor = VehicleSensor(lidar_data, lidar_ids, beliefs[1:length(lidar_ids)])

        # sim wrapper + nearby-human query (this hits belief/planner plumbing too)
        sim = NavigationSimulator(ENV, veh, VEHICLE_PARAMS, sensor, humans, hparams, DT)
        nbh = get_nearby_humans(sim, NUM_NEARBY_HUMANS, MIN_SAFE_DIST, CONE_HALF_ANGLE)

        # touch the same globals your loop writes to (type-stable, fast)
        t_store = round(r*DT, digits=1)
        VEHICLE_TRAJ[t_store] = veh
        BELIEF_ARRAY[t_store] = (nbh.position_data, nbh.ids, nbh.belief)
    end

    return nothing
end

"""
run_loop()

Main outer loop:
  • aligns to the current 0.5 s block key,
  • runs 5 × 0.1 s inner steps: propagate → sense → filter → nearby → publish,
  • records breakdown timings, and sleeps to maintain DT pacing.
"""
function run_loop()
    # Set initial state
    curr_veh = Vehicle(2.0, 2.0, pi/4, 0.0)
    start_time = time()
    j = 0
    while j < 15
        iter_t0 = time()
        println("block_start_time= $(iter_t0-start_time)")

        header_t0    = time()
        elapsed_time = header_t0 - start_time

        hdr_calc_t0 = time()
        t_rel_block = round(elapsed_time, digits=3)
        key_time = round(floor(t_rel_block / 0.5) * 0.5, digits =1)  # use action from last 0.5s mark
        action = get(ACTION_ARRAY, key_time, (0.0, 0.0))
        hdr_calc_t1 = time()
        hdr_calc_s  = hdr_calc_t1 - hdr_calc_t0

        hdr_print_t0 = time()
        hdr_print_t1 = time()
        hdr_print_s  = hdr_print_t1 - hdr_print_t0
        header_s = (hdr_calc_s + hdr_print_s)

        breaks = Vector{NTuple{6,Float64}}(undef, 6)
        steps_total = 0.0
        max_step    = 0.0

        before_for = time() - iter_t0
        println("before_for_loop_time=$before_for")

        for i in 1:5
            loop_start = time()
            counter = 0
            t_loop = round(key_time + (i-1)*DT, digits=1)
            t_store = round(t_loop + 0.1, digits = 1)

            # Propagate vehicle
            t0 = time()
            nx, ny, nθ = move_vehicle(curr_veh.x, curr_veh.y, curr_veh.theta,
                                      VEHICLE_PARAMS.wheelbase, action[1], action[2], DT)
            println("current_time_$counter: ", time()- loop_start); counter +=1
            curr_veh = Vehicle(nx, ny, nθ, action[2])
            println("current_time_$counter: ", time()- loop_start); counter +=1
            VEHICLE_TRAJ[t_store] = curr_veh
            t1 = time()

            # Get latest pedestrian data (from PeopleListener)
            msg = PeopleListener.latest[]
            # NOTE: code uses `veh` (undefined). Likely intended `curr_veh`.
            humans, ids, beliefs = PeopleListener.get_current_humans_and_params(veh, GOALS)
            t2 = time()
            println("current_time_$counter: ", time()- loop_start); counter +=1

            # Pre-filter via LiDAR
            lidar_data, lidar_ids = get_lidar_data_and_ids(veh, humans,
                [HumanParameters(id, [h], 1) for (h, id) in zip(humans, ids)], LIDAR_RANGE)
            lidar_beliefs = beliefs[1:length(lidar_ids)]
            t3 = time()
            println("current_time_$counter: ", time()- loop_start); counter +=1
            
            sensor = VehicleSensor(lidar_data, lidar_ids, lidar_beliefs)
            sim = NavigationSimulator(ENV, veh, VEHICLE_PARAMS, sensor, humans, HumanParameters[], DT)
            t4 = time()
            println("current_time_$counter: ", time()- loop_start); counter +=1

            nbh = get_nearby_humans(sim, NUM_NEARBY_HUMANS, MIN_SAFE_DIST, CONE_HALF_ANGLE)
            BELIEF_ARRAY[t_store] = (nbh.position_data, nbh.ids, nbh.belief)  
            t5 = time()
            println("current_time_$counter: ", time()- loop_start); counter +=1

            publish_live_data(veh, nbh) 
            t6 = time()
            println("current_time_$counter: ", time()- loop_start); counter +=1

            move_s    = t1 - t0
            ppl_s     = t2 - t1
            lidar_s   = t3 - t2
            sim_s     = t4 - t3
            nbh_s     = t5 - t4
            publish_s = t6 - t5
            step_s    = t6 - loop_start

            steps_total += step_s
            max_step     = step_s > max_step ? step_s : max_step
            println("current_time_$counter: ", time()- loop_start); counter +=1

            breaks[i] = (move_s, ppl_s, lidar_s, sim_s, nbh_s, publish_s)

            # pacing to 0.1 s
            now = time() - loop_start
            println("now= $now")
            println("current_time_$counter: ", time()- loop_start); counter +=1
            sleep(max(0.0, 0.09 - now))      # note: 0.09 target here
            println("current_time_$counter: ", time()- loop_start); counter +=1
        end

        block_time = time()
        println("block_end_time= $(block_time-start_time)")
        iter_s = block_time - iter_t0
        if iter_s > 0.5
            @warn "[vehicle_node] TICK overrun (expected 0.5 s)" key_time=key_time iter_s=round(iter_s; digits=6) header_s=round(header_s; digits=6) hdr_calc_s=round(hdr_calc_s; digits=6) hdr_print_s=round(hdr_print_s; digits=6) steps_total=round(steps_total; digits=6) max_step=round(max_step; digits=6)
        end

        for i in 1:5
            move, ppl, lidar, simt, nearby, pub = breaks[i]
            @info "breakdown (s)" move_s= round(move; digits=6) ppl_s=round(ppl; digits=6) lidar_s=round(lidar; digits=6) sim_s=round(simt; digits=6) nearby_s=round(nearby; digits=6) publish_s=round(pub; digits=6)
        end
        j += 1
    end

    println("\n======  run_loop finished  ======")
    vt_keys = sort!(collect(keys(VEHICLE_TRAJ)))
    for t in vt_keys
        veh = VEHICLE_TRAJ[t]
        println("t = $(round(t, digits=1)) "
                * "(x=$(round(veh.x, digits=2)) "
                * "y=$(round(veh.y, digits=2)) "
                * "θ=$(round(veh.theta, digits=2)) "
                * "v=$(round(veh.v, digits=2)))")
    end

    if isempty(ACTION_ARRAY)
        println("\nACTION_ARRAY is empty.")
    else
        aa_keys = sort!(collect(keys(ACTION_ARRAY)))
        println("\nACTION_ARRAY keys → ", aa_keys)
        for t in aa_keys
            s, v = ACTION_ARRAY[t]
            if t isa Integer
                println("tick = ", t, "  t=", round(t*0.5, digits=1),
                        "s  (steering=", round(s, digits=3),
                        ", velocity=", round(v, digits=3), ")")
            else
                println("t = ", round(Float64(t), digits=1),
                        "s  (steering=", round(s, digits=3),
                        ", velocity=", round(v, digits=3), ")")
            end
        end
    end

    println("=================================\n")
end

"""
wait_with_timeout(cond::Condition, timeout::Float64) -> Bool

Waits on `cond` until signaled or `timeout` seconds elapse.
Returns true if signaled, false on timeout.
"""
function wait_with_timeout(cond::Condition, timeout::Float64)
    result = Ref(false)
    task = @async begin
        wait(cond)
        result[] = true
    end
    t_start = time()
    while time() - t_start < timeout
        if result[]; return true; end
        sleep(0.01)
    end
    return false
end

_as_float(x) = x isa AbstractString ? parse(Float64, x) : float(x)
_as_int(x)   = x isa Integer ? Int(x) :
               (x isa AbstractString ? parse(Int, x) : parse(Int, string(x)))

"""
load_traj_map_from_json(json_path; debug=false) -> Dict{Int,Vector{HumanState}}

Parses a JSON array of agents with fields:
  id: Int, path: [ {x, y, v?, goal?: {x,y}} ... ]
Builds a map from agent id → sequence of HumanState.
"""
function load_traj_map_from_json(json_path::AbstractString; debug::Bool=false)
    raw = JSON.parsefile(json_path)
    traj_map = Dict{Int, Vector{HumanState}}()

    @assert raw isa Vector "Expected a top-level JSON array of agents"

    for (ai, agent) in enumerate(raw)
        agent isa Dict || continue
        haskey(agent, "path") || (debug && @warn "[json] agent $ai has no 'path'"; continue)
        id = haskey(agent, "id") ? _as_int(agent["id"]) : (debug && @warn "[json] agent $ai missing 'id'"; continue)
        path = agent["path"]
        path isa Vector || (debug && @warn "[json] 'path' of agent $id is not an array"; continue)

        vec = get!(traj_map, id, HumanState[])

        for (si, s) in enumerate(path)
            try
                x = haskey(s, "x") ? _as_float(s["x"]) : (debug && @warn "[json] agent=$id sample=$si missing x"; continue)
                y = haskey(s, "y") ? _as_float(s["y"]) : (debug && @warn "[json] agent=$id sample=$si missing y"; continue)
                v = haskey(s, "v") ? _as_float(s["v"]) : 0.0
                goal_loc = if haskey(s, "goal") && s["goal"] isa Dict &&
                              haskey(s["goal"], "x") && haskey(s["goal"], "y")
                    Location(_as_float(s["goal"]["x"]), _as_float(s["goal"]["y"]))
                else
                    Location(0.0, 0.0)
                end
                push!(vec, HumanState(x, y, v, goal_loc))
            catch e
                debug && @warn "[json] skipping agent=$id sample=$si due to $e"
            end
        end
        debug && println("[json] agent id=$id  samples=", length(vec))
    end

    @assert !isempty(traj_map) "No trajectories parsed; check JSON structure/path"
    return traj_map
end

human_uniform_belief(n::Integer) = HumanGoalsBelief(fill(1.0 / max(1,n), max(1,n)))

"""
populate_sim_objects!(out, veh_traj_dict, traj_map, env, veh_params, env_humans_params;
                      dt=0.1, ped_dt=0.01, hold_last=true)

Builds an OrderedDict of NavigationSimulator objects across vehicle trajectory
timestamps, aligning pedestrian samples by downsampling ratio dt/ped_dt.
"""
function populate_sim_objects!(
        out::Output{P},
        veh_traj_dict::Dict{Float64,Vehicle},
        traj_map::Dict{Int,Vector{HumanState}},
        env::ExperimentEnvironment,
        veh_params::VehicleParametersESPlanner,
        env_humans_params::Vector{HumanParameters};
        dt::Float64=0.1, ped_dt::Float64=0.01, hold_last::Bool=true
    ) where {P}

    out.sim_objects = OrderedDict{Float64,NavigationSimulator{P}}()

    step_ratio = round(Int, dt / ped_dt)
    ped_ids    = sort!(collect(keys(traj_map)))
    downsampled = Dict(id => traj_map[id][1:step_ratio:end] for id in ped_ids)

    sorted_times = sort!(collect(keys(veh_traj_dict)))
    for (idx, t) in enumerate(sorted_times)
        vehicle = veh_traj_dict[t]

        humans = HumanState[]
        for id in ped_ids
            vec = downsampled[id]
            if idx ≤ length(vec)
                push!(humans, vec[idx])
            elseif hold_last
                push!(humans, vec[end])
            end
        end

        ids         = ped_ids
        num_goals   = length(get_human_goals(env))
        beliefs     = [human_uniform_belief(num_goals) for _ in 1:length(ped_ids)]
        sensor      = VehicleSensor(humans, ids, beliefs)

        sim = NavigationSimulator{Any}(env, vehicle, veh_params, sensor,
                                       humans, env_humans_param
