using Random
using DataStructures
using Parameters
using LazySets

#Various different Struct definitions

struct Location
    x::Float64
    y::Float64
end

struct ObstacleLocation
    x::Float64
    y::Float64
    r::Float64 #Radius of the obstacle which is assumed to be a circle
end

#=
Used to store belief over goals for a single human
Need this for every human
=#
struct HumanGoalsBelief
    pdf::Array{Float64,1}
end

struct HumanState
    x::Float64
    y::Float64
    v::Float64
    goal::Location
end

mutable struct HumanParameters
    id::Int64
    path::Array{HumanState,1}
    path_index::Int64
end

struct NearbyHumans
    position_data::Array{HumanState,1}
    ids::Array{Int64,1}
    belief::Array{HumanGoalsBelief,1}
end
# nbh = NearbyHumans(HumanState[], Int64[], HumanGoalsBelief[])

struct Vehicle
    x::Float64
    y::Float64
    theta::Float64
    v::Float64
end
# v = Vehicle(0.0,0.0,0.0,0.0)

struct VehicleSensor
    lidar_data::Array{HumanState,1}
    ids::Array{Int64,1}
    belief::Array{HumanGoalsBelief,1}
end
# vsd = VehicleSensor(HumanState[], Int64[], HumanGoalsBelief[])

struct VehicleParametersESPlanner
    wheelbase::Float64
    length::Float64
    breadth::Float64
    dist_origin_to_center::Float64
    radius::Float64
    max_speed::Float64
    max_steering_angle::Float64
    goal::Location
end
# vp_es = VehicleParametersESPlanner(0.3,3.0,Location(100.0,100.0))

struct VehicleParametersLSPlanner
    wheelbase::Float64
    length::Float64
    breadth::Float64
    dist_origin_to_center::Float64
    radius::Float64
    max_speed::Float64
    max_steering_angle::Float64
    goal::Location
    controls_sequence::Array{Float64,1}
end
# vp_ls = VehicleParametersLSPlanner(0.3,3.0,Location(100.0,100.0),Float64[])

struct ExperimentEnvironment
    length::Float64
    breadth::Float64
    obstacles::Array{ObstacleLocation,1}
end
#e = ExperimentEnvironment(100.0,100.0,ObstacleLocation[])

mutable struct ExperimentDetails
    user_defined_rng::AbstractRNG
    veh_path_planning_v::Float64
    num_humans_env::Int64
    human_start_v::Float64
    one_time_step::Float64
    simulator_time_step::Float64
    lidar_range::Float64
    MAX_TIME_LIMIT::Float64
    min_safe_distance_from_human::Float64
    radius_around_vehicle_goal::Float64
    max_risk_distance::Float64
    update_sensor_data_time_interval::Float64
    buffer_time::Float64
    human_goal_locations::Array{Location,1}
    env::ExperimentEnvironment
end

mutable struct PathPlanningDetails
    radius_around_uncertain_human::Float64
    min_safe_distance_from_human::Float64
    human_collision_cost::Float64
    num_human_goals::Int64
    human_goals::Array{Location,1}
    veh_path_planning_v::Float64
    radius_around_vehicle_goal::Float64
    lambda::Float64
    one_time_step::Float64
    planning_time::Float64
end

mutable struct POMPDPlanningDetails
    num_nearby_humans::Int64
    cone_half_angle::Float64
    min_safe_distance_from_human::Float64
    min_safe_distance_from_obstacle::Float64
    radius_around_vehicle_goal::Float64
    human_collision_penalty::Float64
    obstacle_collision_penalty::Float64
    goal_reached_reward::Float64
    max_vehicle_speed::Float64
    max_vehicle_steering_angle::Float64
    num_segments_in_one_time_step::Int64
    observation_discretization_length::Float64
    d_near::Float64
    d_far::Float64
    planning_time::Float64
    max_num_trials::Int64
    action_delta_speed::Float64
    action_max_delta_heading_angle::Float64
    tree_search_max_depth::Int64
    num_scenarios::Int64
    discount_factor::Float64
    one_time_step::Float64
end

struct HJBPlanningDetails
    Dt::Float64
    max_solve_steps::Int64
    Dval_tol::Float64
    max_vehicle_steering_angle::Float64
    max_vehicle_speed::Float64
end

@with_kw mutable struct InputParameters
    rng::AbstractRNG = MersenneTwister(rand(UInt32))
    env_length::Float64 = 5.518
    env_breadth::Float64 = 11.036
    obstacles::Array{ObstacleLocation,1} = ObstacleLocation[]
    num_humans_env::Int64 = 200
    human_start_v::Float64 = 1.0
    veh_start_x::Float64 = 2.75
    veh_start_y::Float64 = 0.5
    veh_start_theta::Float64 = 0.0
    veh_start_v::Float64 = 0.0
    veh_goal_x::Float64 = 2.759
    veh_goal_y::Float64 = 11.0
    veh_wheelbase::Float64 = 0.3
    veh_length::Float64 = 0.5207
    veh_breadth::Float64 = 0.2762
    veh_dist_origin_to_center::Float64 = 0.1715
    veh_max_speed::Float64 = 2.0
    veh_max_steering_angle = 0.475
    veh_path_planning_v::Float64 = 1.0
    lidar_range::Float64 = 30.0
    num_nearby_humans::Int64 = 6
    cone_half_angle::Float64 = 2*pi/3
    pomdp_discount_factor::Float64 = 0.97
    min_safe_distance_from_human::Float64 = 1.0
    min_safe_distance_from_obstacle::Float64 = 1.0
    radius_around_vehicle_goal::Float64 = 1.0
    max_risk_distance::Float64 = 0.5
    human_collision_penalty::Float64 = -100.0
    obstacle_collision_penalty::Float64 = -100.0
    goal_reached_reward::Float64 = 1000.0
    num_segments_in_one_time_step::Int64 = 10
    observation_discretization_length::Float64 = 1.0
    tree_search_max_depth::Int64 = 100
    num_scenarios::Int64 = 100
    d_near::Float64 = 0.5
    d_far::Float64 = 0.5
    ES_pomdp_planning_time::Float64 = 0.4
    ES_max_num_trials::Int64 = 11
    LS_pomdp_planning_time::Float64 = 0.3
    LS_max_num_trials::Int64 = 101
    pomdp_action_max_delta_heading_angle::Float64 = pi/4
    pomdp_action_delta_speed::Float64 = 0.5
    one_time_step::Float64 = 0.5
    simulator_time_step::Float64 = 0.1
    update_sensor_data_time_interval::Float64 = 0.1
    buffer_time::Float64 = 0.2
    MAX_TIME_LIMIT::Float64 = 200.0
end

mutable struct Output{T}
    number_sudden_stops::Int64
    number_risky_scenarios::Int64
    time_taken::Float64
    vehicle_body::T
    vehicle_ran_into_boundary_wall::Bool
    vehicle_ran_into_obstacle::Bool
    vehicle_reached_goal::Bool
    vehicle_expected_trajectory::OrderedDict
    pomdp_planners::OrderedDict
    nearby_humans::OrderedDict
    b_root::OrderedDict
    despot_trees::OrderedDict
    vehicle_actions::OrderedDict
    sim_objects::OrderedDict
    risky_scenarios::OrderedDict
end
