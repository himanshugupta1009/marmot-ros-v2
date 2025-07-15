#!/usr/bin/env julia
# vehicle_client.jl - Client only  
using RobotOS
@rosimport dummy_data_pkg.msg: PeoplePoseArray
@rosimport dummy_data_pkg.srv: MoveVehicle
rostypegen()
using .dummy_data_pkg.msg: PeoplePoseArray
using .dummy_data_pkg.srv: MoveVehicle, MoveVehicleRequest, MoveVehicleResponse

include(joinpath(@__DIR__, "struct_definition.jl"))

const ACTION_STEERING = 0.0
const ACTION_SPEED = 1.0
const DT = 0.1
const loop_time = 0.5 
const N_CYCLES = 10

const veh_traj_dict = Dict{Int,Vehicle}()               # vehicle position
const action_dict   = Dict{Int,Tuple{Float64,Float64}}()    # Action (speed, steering)

const latest_people = Ref(PeoplePoseArray())
people_cb(msg) = (latest_people[] = msg)

function predict_vehicle_state(veh::Vehicle)
    wait_for_service("/move_vehicle")
    srv = ServiceProxy{MoveVehicle}("/move_vehicle")

    # ---- build request --------------------------------------------------
    req = MoveVehicleRequest()
    req.x, req.y, req.theta, req.v = veh.x, veh.y, veh.theta, veh.v
    req.steering = ACTION_STEERING
    req.speed    = ACTION_SPEED
    req.dt       = DT
    # ---------------------------------------------------------------------

    resp = srv(req)                         # single return value
    #return Vehicle(resp.xs, resp.ys, resp.thetas, resp.vs)
    return resp
end

function run_controller()
    veh = Vehicle(0.0, 0.0, 0.0, 0.0)
    @info "Client loop starting" cycles=N_CYCLES
    planning_rate = Rate(1/loop_time)
     #Time convertion to Hz
    inner_loop_counter = loop_time/DT
    global_idx = 0

    for k in 1:N_CYCLES 
        t0 = time() # Has to be the first
        action_dict[k] = (ACTION_STEERING, ACTION_SPEED)
        # global_idx += 1  # dict indexing counter
        resp = predict_vehicle_state(veh)
        println(resp)
        for i in eachindex(resp.xs)
            global_idx += 1
            veh = Vehicle(resp.xs[i], resp.ys[i], resp.thetas[i], resp.vs[i])
            veh_traj_dict[global_idx] = veh
        end
        @info "cycle $k → vehicle" veh
        @info "going to sleep"
        sleep(planning_rate)
    end
    @info "Finished $N_CYCLES cycles"
    @info "veh_traj_dict" veh_traj_dict
    @info "action_dict"   action_dict
end

function main()
    RobotOS.init_node("vehicle_client"; anonymous=true)
    Subscriber("/car/dummy/people_poses", PeoplePoseArray, people_cb)
    @async RobotOS.spin()             # background callbacks
    sleep(1.0)                        # give ROS a moment
    run_controller()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end