#!/usr/bin/env julia
#
# merge_logs.jl  –  combine vehicle and pedestrian logs into Output.sim_objects
#
##########################################################################################
import Pkg; Pkg.activate(joinpath(@__DIR__))
using Serialization
using DataStructures: OrderedDict
using Printf: @printf
using Plots, LazySets, StaticArrays
using RobotOS

##############  bring in your type definitions + env / params used during the run  ########
include(joinpath(@__DIR__, "..", "src", "struct_definition.jl"))
include(joinpath(@__DIR__, "people_listener.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src", "main_es.jl"))   # provides `env`,`vehicle_params`,`humans_params`
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src", "visualization.jl"))
include(joinpath(@__DIR__, "..", "..", "..","..", "src", "human_aware_navigation_modifiedbyansh", "src", "belief_tracker.jl"))
# @rosimport dummy_data_pkg.msg: PeoplePoseArray
# RobotOS.rostypegen()
#using .dummy_data_pkg.msg: PeoplePoseArray

const rng            = Main.exp_details.user_defined_rng
const MAX_TIME_LIMIT = Main.exp_details.MAX_TIME_LIMIT

const NUM_NEARBY_HUMANS   = 6          # same as input_config.num_nearby_humans
const MIN_SAFE_DIST       = 1.0        # metres (front-cone exception)
const CONE_HALF_ANGLE     = 2*pi/3       # 120° front field-of-view
const LIDAR_RANGE         = 20.0
# const veh_params = Main.veh_params
#const env_humans_params = Main.env_humans_params
##########################################################################################

#──────────────────────── helper: build sim_objects at 0.1-s resolution ──────────────────
"""
    populate_sim_objects!(
        out, veh_traj_dict, traj_map, env, veh_params, env_humans_params;
        dt      = 0.1,    # vehicle Δt
        ped_dt  = 0.01,   # pedestrian Δt
        hold_last = true  # repeat last pose if a ped ends early
    )

Fills `out.sim_objects` (OrderedDict{Float64,NavigationSimulator}) by aligning

* `veh_traj_dict`  – Dict{Float64,Vehicle}   at `dt` resolution
* `traj_map`       – Dict{Int,Vector{HumanState}} at `ped_dt` resolution

It keeps every `round(Int, dt/ped_dt)`-th pedestrian sample so vehicle and
pedestrian snapshots share the same timeline.
"""

## THIS FUNCTION WORKS PERFECTLY WITHOUT ANY BELIEF UPDATE
function populate_sim_objects!(
        out::Output{P},
        veh_traj_dict::Dict{Float64,Vehicle},
        traj_map::Dict{Int,Vector{HumanState}},
        env::ExperimentEnvironment,
        # vehicle_body::P,
        veh_params::VehicleParametersESPlanner,
        env_humans_params::Vector{HumanParameters};
        dt::Float64=0.1,
        ped_dt::Float64=0.01,
        hold_last::Bool=true
    ) where {P}

    out.sim_objects = OrderedDict{Float64,NavigationSimulator{P}}()

    step_ratio = round(Int, dt / ped_dt)         # 10 for 0.1 / 0.01
    ped_ids    = sort!(collect(keys(traj_map)))

    downsampled = Dict(id => traj_map[id][1:step_ratio:end] for id in ped_ids)

    sorted_times = sort!(collect(keys(veh_traj_dict)))   # Vector{Float64}
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

        ids     = ped_ids                                 # 1-to-1 with `humans`
        num_goals = length(get_human_goals(env))          # uniform belief over all goals
        beliefs = [default_human_goal_belief(num_goals) for _ in ids]
        sensor  = VehicleSensor(humans, ids, beliefs)
        
        sim = NavigationSimulator{Any}(
            env,
            vehicle,
            veh_params,
            sensor,   
            humans,
            env_humans_params,
            dt
        )
        out.sim_objects[t] = sim
    end

    return out
end


function print_belief_snapshot(t, sensor; max_lines = 6, digits = 3)
    println("[Belief snapshot @ t = $(round(t, digits = 2)) s]")
    n = min(max_lines, length(sensor.ids))
    for k in 1:n
        id = sensor.ids[k]
        h  = sensor.lidar_data[k]                     # HumanState
        b  = sensor.belief[k].pdf     # Vector of Float64
        println("Human $(id) @ (x=$(round(h.x, digits=digits)), " *
                "y=$(round(h.y, digits=digits))) belief = " *
                string(round.(b; digits = digits)))
    end
    println()
end


#──────────────────────────────────────────────────────────────────────────────────────────

println("[merge]  loading logs …")
veh_traj_dict = deserialize("veh_log.jls") :: Dict{Float64,Vehicle}

to_Main_location(loc::PeopleListener.Location) =
    Location(loc.x, loc.y)

to_Main_state(h::PeopleListener.HumanState) =
    HumanState(h.x, h.y, h.v,
               to_Main_location(h.goal))      # <-- convert the nested struct
raw_traj = deserialize("ped_log.jls")    # Dict{Int,Vector{PeopleListener.HumanState}}

traj_map = Dict(
    id => [to_Main_state(h) for h in vec]
    for (id, vec) in raw_traj
)

println("[merge]  vehicle poses      : ", length(veh_traj_dict))
println("[merge]  pedestrian IDs     : ", length(traj_map))

# println("\n[ped_log] ---- initial entries (raw) ----")
# num_ids_to_show      = 7        # change if you want more / fewer
# num_samples_per_id   = 11

# for (idx, (pid, vec)) in enumerate(raw_traj)    # idx = 1,2,3…
#     idx > num_ids_to_show && break

#     println("  id = ", pid,
#             "   (showing first ", num_samples_per_id, " samples)")
#     for i in 1:min(num_samples_per_id, length(vec))
#         h = vec[i]
#         @printf("      sample %2d: (x=%.3f, y=%.3f)\n",
#                 i, h.x, h.y)
#     end
# end
# println("[ped_log] ----------------------------------\n")
#────────────────────────────────────────────────────────────────

###########################  build a new Output container  ################################
output = Output{Any}(
    0, 0, 0.0,
    # vehicle_body,   # <-- use your actual VehicleBody ctor if different
    nothing,
    false, false, false,
    OrderedDict(), OrderedDict(), OrderedDict(),
    OrderedDict(), OrderedDict(), OrderedDict(),
    OrderedDict(), OrderedDict()
)
###########################################################################################

populate_sim_objects!(output, veh_traj_dict, traj_map,
                      env, Main.veh_params, env_humans_params)
println("[merge]  produced ", length(output.sim_objects), " snapshots")

const PeoplePoseArray = PeopleListener.PeoplePoseArray
PeopleListener.latest[] = PeoplePoseArray()

nearby_dict = OrderedDict{Float64,NearbyHumans}()

for (t, sim) in output.sim_objects
    #Pre-Filter by lidar range
    lidar_data, ids = get_lidar_data_and_ids(sim.vehicle,
                                             sim.humans,
                                             sim.humans_params,
                                             LIDAR_RANGE)
    belief = sim.vehicle_sensor_data.belief[1:length(ids)]

    # sim.vehicle_sensor_data.lidar_data = lidar_data
    # sim.vehicle_sensor_data.ids        = ids
    tmp_sensor = VehicleSensor(lidar_data, ids, belief)

    tmp_sim = NavigationSimulator(sim.env,
                                  sim.vehicle,
                                  sim.vehicle_params,
                                  tmp_sensor,          # ← new sensor
                                  sim.humans,
                                  sim.humans_params,
                                  sim.one_time_step)

    nbh = get_nearby_humans(tmp_sim,
                             NUM_NEARBY_HUMANS,
                             MIN_SAFE_DIST,
                             CONE_HALF_ANGLE)
    nearby_dict[t] = nbh
end
println("[post processing] computed nearby-human sets for ", length(nearby_dict), " frames")
@show first(values(nearby_dict)).ids

# for (k, (t, sim)) in enumerate(output.sim_objects)
#     if k > 3                       # only show the first three
#         break
#     end
#     println("\n  t = ", round(t, digits = 2), " s")
#     # vehicle pose
#     vp = sim.vehicle
#     @printf("    vehicle  (x, y, θ, v) = (%.3f, %.3f, %.2f°, %.2f m/s)\n",
#             vp.x, vp.y, vp.theta * 180/π, vp.v)

#     # humans
#     num_peds = length(sim.humans)
#     print("    ", num_peds, " pedestrian")
#     println(num_peds == 1 ? ":" : "s:")
#     for (idx, h) in enumerate(sim.humans)      # idx = 1,2,3… or treat as pid
#         pid = idx                              # or  pid = h.id  if that field exists
#         @printf("        id=%d  (x, y)=(%.3f, %.3f) \n",
#                 pid, h.x, h.y)
#     end
# end
# println("[merge]  -----------------------------------------------\n")

serialize("sim_objects.jls", output.sim_objects)
println("[merge]  wrote sim_objects.jls")


sim_objects = output.sim_objects

create_gif = true
# create_gif = false

if(create_gif)

    vehicle_body = VPolygon([SVector(0.0, 0.0)])

    # *** ONLY the fields that get_plot()/observe() actually read ***
    exp_details = ExperimentDetails(
        rng,                       # keep your existing RNG
        1.0,                       # veh_path_planning_v (dummy)
        0,                         # num_humans_env      (dummy)
        1.0,                       # human_start_v       (dummy)
        0.5, 0.1, 30.0, MAX_TIME_LIMIT,
        1.0, 1.0, 0.5,
        0.1, 0.2,
        get_human_goals(env),      # real goals
        env                        # real environment
    )

    empty_dict() = OrderedDict{Float64,Any}()
    gif_output = Output(
        0, 0, 0.0,                 # counters
        vehicle_body, false, false, false,
        OrderedDict(),           # vehicle_expected_trajectory (not needed for gif)
        OrderedDict(),           # pomdp_planners
        nearby_dict,             # <<<––– real nearby humans
        OrderedDict(),           # b_root
        OrderedDict(),           # despot_trees
        OrderedDict(),           # vehicle_actions
        sim_objects,             # sim_objects already built
        empty_dict()           # risky_scenarios
    )

    # 3.  Generate and save the animation
    ##########################################################################
    generate_gif(gif_output, exp_details)   # writes es_planner.gif
    @info "merge_logs" "Wrote es_planner.gif "
end

