#!/usr/bin/env julia
# =============================================================================
# File: realtime_vehicle_client.jl
# Role:
#   Planner-side ROS client that:
#     • waits for /car/sim/GetLiveData and /set_action services,
#     • pulls the latest LiveData snapshot,
#     • builds a planning scenario,
#     • computes/samples an action, and
#     • schedules it via /set_action for the next 0.5 s block.
#
# ROS Interfaces (consumed):
#   Service  : /car/sim/GetLiveData   (dummy_data_pkg/GetLiveData)
#     -> returns snaps::Vector{LiveData}; this client uses the newest [1]
#   Service  : /set_action            (dummy_data_pkg/SetAction)
#     -> request: (t_exec::Float64, steering_angle::Float64, delta_speed::Float64)
#
# Data Contracts (used fields in LiveData):
#   • Vehicle (Float32): x, y, theta, v
#   • Humans:
#       ids::Vector{Int32}
#       hx, hy::Vector{Float32}        # human positions (z assumed 0 here)
#       belief_pdf::Vector{Float32}    # concatenated per-human over goals
#
# Timing:
#   • TOTAL_TIME = 0.5 s (action cadence)
#   • DT         = 0.1 s (inner sim step in vehicle node; used for context)
#   • key_time   increments by 0.5 s each iteration. Actions are scheduled at
#     t_exec = key_time + TOTAL_TIME (i.e., for the NEXT block).
#
# Key Variables (high level):
#   • env, veh_params, exp_details      — environment and vehicle configuration
#   • pomdp_details, pomdp, solver      — planning configuration and solver
#   • GLD, SA                           — ServiceProxy handles for LiveData/SetAction
#   • possible_speeds, possible_angles  — discrete action candidates (if sampling)
#
# Gotchas:
#   • Ensure /car/sim/LiveData is being published before tight loops; this
#     client calls wait_for_first_live(GLD) to block until a valid snapshot.
#   • Float time keys: /set_action may expect exact t_exec alignment (0.5 s grid).
#   • Paths in include(...) assume repo layout under catkin_ws/src; adjust as needed.
# =============================================================================

import Pkg; Pkg.activate(joinpath(@__DIR__))
using RobotOS
@rosimport dummy_data_pkg.msg: PeoplePoseArray, LiveData, BeliefArray
@rosimport dummy_data_pkg.srv: SetAction, GetLiveData
@rosimport std_msgs.msg: Header
rostypegen()
using .dummy_data_pkg.msg: PeoplePoseArray, LiveData, BeliefArray
using .dummy_data_pkg.srv: SetAction, SetActionRequest, GetLiveData, GetLiveDataRequest, GetLiveDataResponse
using BellmanPDEs
using JLD2
using Random
using .std_msgs.msg: Header
import Dates

# --- Project includes (paths tailored to this workspace) ----------------------
include(joinpath(@__DIR__, "struct_definition.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","environment.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","simulator.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","ES_POMDP_Planner.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","belief_tracker.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","HJB_wrappers.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","parser.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","simulator_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","configs", "small_obstacles_50x50.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","visualization.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","shielding/shield_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src","shielding/shield.jl"))

# --- Environment / experiment setup ------------------------------------------
env = generate_environment(input_config.env_length, input_config.env_breadth, input_config.obstacles)
exp_details = ExperimentDetails(
    input_config.rng,
    input_config.veh_path_planning_v,
    input_config.num_humans_env,
    input_config.human_start_v,
    input_config.one_time_step,
    input_config.simulator_time_step,
    input_config.lidar_range,
    input_config.MAX_TIME_LIMIT,
    input_config.min_safe_distance_from_human,
    input_config.radius_around_vehicle_goal,
    input_config.max_risk_distance,
    input_config.update_sensor_data_time_interval,
    input_config.buffer_time,
    get_human_goals(env),          # human_goal_locations
    env                            # env
)

veh_goal = Location(input_config.veh_goal_x, input_config.veh_goal_y)
r = sqrt((0.5*input_config.veh_length)^2 + (0.5*input_config.veh_breadth)^2)
veh_params = VehicleParametersESPlanner(input_config.veh_wheelbase, input_config.veh_length,
    input_config.veh_breadth, input_config.veh_dist_origin_to_center, r,
    input_config.veh_max_speed, input_config.veh_max_steering_angle, veh_goal)
vehicle_body = get_vehicle_body((veh_params.length,veh_params.length), (veh_params.dist_origin_to_center,0.0))

# --- Rollout guide / POMDP setup ---------------------------------------------
rollout_path = expanduser("~/catkin_ws/src/human_aware_navigation_modifiedbyansh/src/rollout_guides/HJB_rollout_guide_no_obstacles_50x50.jld2")
rollout_guide = JLD2.load(rollout_path, "rollout_guide")

custom_pomdp_planning_time = 0.4  # if you want to override defaults
sol_rng = MersenneTwister(19)

pomdp_details = POMPDPlanningDetails(
    input_config.num_nearby_humans,
    input_config.cone_half_angle,
    input_config.min_safe_distance_from_human,
    input_config.min_safe_distance_from_obstacle,
    input_config.radius_around_vehicle_goal,
    input_config.human_collision_penalty,
    input_config.obstacle_collision_penalty,
    input_config.goal_reached_reward,
    input_config.veh_max_speed,
    input_config.veh_max_steering_angle,
    input_config.num_segments_in_one_time_step,
    input_config.observation_discretization_length,
    input_config.d_near,
    input_config.d_far,
    input_config.ES_pomdp_planning_time,
    input_config.ES_max_num_trials,
    input_config.pomdp_action_delta_speed,
    input_config.pomdp_action_max_delta_heading_angle,
    input_config.tree_search_max_depth,
    input_config.num_scenarios,
    input_config.pomdp_discount_factor,
    input_config.one_time_step)

pomdp = ExtendedSpacePOMDP(pomdp_details, env, veh_params, rollout_guide, true)

T_max = true
solver = DESPOTSolver(
        bounds=IndependentBounds(
        DefaultPolicyLB(FunctionPolicy(b -> calculate_lower_bound(pomdp, b)),
        max_depth=pomdp_details.tree_search_max_depth),
        old_calculate_upper_bound,
        check_terminal=true, 
        consistency_fix_thresh=1e-5),
    K=pomdp_details.num_scenarios, 
    D=pomdp_details.tree_search_max_depth,
    tree_in_info=true,
    T_max= false ? Inf : pomdp_details.planning_time,
    max_trials = false ? pomdp_details.max_num_trials : 100,
    default_action=get_default_action,
    rng=sol_rng
)
planner = POMDPs.solve(solver, pomdp)

println("[setup]   environment and planner ready")

# --- Core timing / geometry knobs --------------------------------------------
const DT = exp_details.simulator_time_step        # expected 0.1 s
const TOTAL_TIME = 0.5                             # action horizon (s)
const N_STEPS = Int(TOTAL_TIME / DT)               # horizon steps (typically 5)

const LIDAR_RANGE = exp_details.lidar_range
const NUM_NEARBY_HUMANS = Main.pomdp_details.num_nearby_humans
const MIN_SAFE_DIST = exp_details.min_safe_distance_from_human
const CONE_HALF_ANGLE = Main.pomdp_details.cone_half_angle
const radius = Main.exp_details.radius_around_vehicle_goal

# Candidate discrete actions (used when sampling)
possible_speeds = [0.0, 0.5, 1.0, 1.5, 2.0]
possible_angles = collect(-10:10)

# -----------------------------------------------------------------------------
"""
snapshot_to_vectors(snaps::LiveData, num_goals::Int)
    -> (humans::Vector{HumanState}, ids::Vector{Int64}, beliefs::Vector{HumanGoalsBelief})

Convert a LiveData snapshot to typed arrays used by the planner.

- Assumes `belief_pdf` is concatenated per-human across `num_goals`.
- Normalizes each human's belief vector to sum to 1.0.

Args:
  snaps::LiveData   LiveData message (vehicle + humans + beliefs)
  num_goals::Int    number of discrete goals per human

Returns:
  Tuple of humans, ids, beliefs aligned by index.
"""
function snapshot_to_vectors(snaps::LiveData, num_goals::Int)
    n = length(snaps.ids)

    humans  = Vector{HumanState}(undef, n)
    ids     = Vector{Int64}(undef, n)
    beliefs = Vector{HumanGoalsBelief}(undef, n)

    @inbounds for i in 1:n
        humans[i] = HumanState(snaps.hx[i], snaps.hy[i], 0.0, Location(0, 0))
        ids[i] = Int(snaps.ids[i])
        offset = (i-1)*num_goals
        pdf_f64 = Float64.(snaps.belief_pdf[offset+1 : offset+num_goals])
        beliefs[i] = HumanGoalsBelief(pdf_f64 ./ sum(pdf_f64))   # normalize
    end
    return humans, ids, beliefs
end

# -----------------------------------------------------------------------------
"""
wait_for_first_live(GLD::ServiceProxy{GetLiveData}; poll=0.05) -> LiveData

Blocks until GetLiveData returns a non-empty snapshot with at least one id.
Useful at startup to ensure downstream code has valid arrays.

Args:
  GLD  : ServiceProxy handle to /car/sim/GetLiveData
  poll : polling sleep (seconds), default 0.05

Returns:
  LiveData (first valid snapshot)
"""
function wait_for_first_live(GLD::ServiceProxy{GetLiveData}; poll::Float64 = 0.05)
    req = GetLiveDataRequest()           # empty request
    while true
        snaps = GLD(req).snaps
        if !isempty(snaps)
            snap = snaps[1]
            if !isempty(snap.ids)
                return snap
            end
        end
        sleep(poll)
    end
end

# -----------------------------------------------------------------------------
"""
warmup_action_info!(planner, env, exp_details, veh_params, horizon)

Pre-JIT common planning paths by constructing a benign scenario with
6 dummy humans and uniform 4-goal beliefs, then calling `action_info`
several times.

Side effect: prints a confirmation upon completion.
"""
function warmup_action_info!(planner, env, exp_details, veh_params, horizon)
    # Vehicle at a benign pose
    veh = Vehicle(env.length/2, env.breadth/2, 0.0, 0.0)

    # Ensure exactly 4 goals for the warmup scenario
    goals_in = exp_details.human_goal_locations
    goals4 = if length(goals_in) >= 4
        goals_in[1:4]
    else
        # Pad with synthetic goals at corners if needed
        base = vcat(
            goals_in,
            [Location(0.0, 0.0),
             Location(env.length, 0.0),
             Location(env.length, env.breadth),
             Location(0.0, env.breadth)]
        )
        base[1:4]
    end

    # Build 6 dummy humans, far away, round-robin assigning one of the 4 goals
    humans = [HumanState(-1e6 - i, -1e6 - i, 0.0, goals4[mod1(i, 4)]) for i in 1:6]

    # Each human has belief [0.25, 0.25, 0.25, 0.25]
    beliefs = [HumanGoalsBelief(fill(0.25, 4)) for _ in 1:6]

    b = TreeSearchScenarioParameters(
        veh.x, veh.y, veh.theta, veh.v,
        modify_vehicle_params(veh_params),
        goals4,                     # exactly 4 goals to match 4-entry beliefs
        length(humans), humans,
        beliefs,
        env.length, env.breadth, horizon
    )

    # Trigger JIT a couple of times
    action_info(planner, b)
    action_info(planner, b)
    action_info(planner, b)
    println("action_info loaded (6 humans, uniform 4-goal belief)")
end

# -----------------------------------------------------------------------------
"""
warmup_once!(SA, env, veh_params, exp_details; num_humans=2)

Optional one-time warmup that constructs a synthetic LiveData message,
mirrors minimal loop work, and sends a harmless SetAction request to JIT
serialization paths. It swallows service errors if the server isn't ready.

Args:
  SA::ServiceProxy{SetAction}
"""
function warmup_once!(SA, env, veh_params, exp_details; num_humans=2)
    num_goals = length(exp_details.human_goal_locations)

    # Build a LiveData msg by field assignment (no keyword ctor)
    snap = LiveData()
    snap.header = Header()
    snap.header.stamp = RobotOS.now()

    # Vehicle state (Float32 fields)
    snap.x = 0.0f0
    snap.y = 0.0f0
    snap.theta = 0.0f0
    snap.v = 0.0f0

    # Humans (types must match: ids Int32[], hx/hy Float32[])
    snap.ids = Int32[1:num_humans...]
    snap.hx  = Float32.(collect(1:num_humans))
    snap.hy  = Float32.(collect(1:num_humans))

    # Belief pdf is concatenated per-human over all goals (Float32[])
    pdf = Float32[]
    uni = Float32.(fill(1/num_goals, num_goals))
    for _ in 1:num_humans
        append!(pdf, uni)
    end
    snap.belief_pdf = pdf

    # Mirror loop work (no printing)
    veh  = Vehicle(snap.x, snap.y, snap.theta, snap.v)
    _dist_to_goal = hypot(veh.x - veh_params.goal.x, veh.y - veh_params.goal.y)

    humans, ids, beliefs = snapshot_to_vectors(snap, num_goals)

    b = TreeSearchScenarioParameters(
        veh.x, veh.y, veh.theta, veh.v,
        modify_vehicle_params(veh_params),
        exp_details.human_goal_locations,
        length(humans), humans, beliefs,
        env.length, env.breadth, exp_details.one_time_step
    )

    # Random action 
    possible_speeds = (0.0, 0.5, 1.0, 1.5, 2.0)
    possible_angles = -10:10
    action = (steering_angle = rand(possible_angles),
              delta_speed    = rand(possible_speeds))

    # Service call warmup (ignore result; it just JITs the path)
    t_exec = exp_details.one_time_step
    req = SetActionRequest(t_exec, action.steering_angle, action.delta_speed)
    try
        SA(req)
    catch _
        # It's okay if service isn't up yet; we only need compilation.
    end
    println("\n Warmup done")

    return nothing
end

# -----------------------------------------------------------------------------
"""
main()

Planner main:
  • init ROS node and spin in background,
  • obtain service proxies to GetLiveData and SetAction,
  • optionally warm up types/serialization,
  • loop at 0.5 s cadence: read snapshot → build scenario → pick action →
    schedule action for the next block.

Prints loop duration diagnostics at the end.
"""
function main()
    println(">>> Entered realtime_planner main()")
    RobotOS.init_node("realtime_planner")
    @async RobotOS.spin()  # let callbacks run if needed later

    println("[planner] waiting for /car/sim/GetLiveData …")
    wait_for_service("/car/sim/GetLiveData")
    GLD = ServiceProxy{GetLiveData}("/car/sim/GetLiveData")
    
    warmup_action_info!(planner, env, exp_details, veh_params, TOTAL_TIME)

    println("[planner] waiting for /set_action …")
    wait_for_service("/set_action")
    SA = ServiceProxy{SetAction}("/set_action")
    println("done")

    # --- warmup (one-time) ---
    try
        println("[planner] Priming GetLiveData…") 
