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
const ACTION_SPEED = 2.0
const DT = 0.1
const loop_time = 0.5 
const N_CYCLES = 72

const veh_traj_dict = Dict{Float64,Vehicle}()               # vehicle position
const action_dict   = Dict{Float64,Tuple{Float64,Float64}}()    # Action (speed, steering)

const latest_people = Ref(PeoplePoseArray())
people_cb(msg) = (latest_people[] = msg)

function predict_vehicle_state(veh::Vehicle, t_req::Float64)
    wait_for_service("/move_vehicle")
    srv = ServiceProxy{MoveVehicle}("/move_vehicle")

    # ---- build request --------------------------------------------------
    req = MoveVehicleRequest()
    req.x, req.y, req.theta, req.v = veh.x, veh.y, veh.theta, veh.v
    req.steering = ACTION_STEERING
    req.speed    = ACTION_SPEED
    req.dt       = DT
    req.t0       = t_req
    # ---------------------------------------------------------------------

    resp = srv(req)                         # single return value
    #return Vehicle(resp.xs, resp.ys, resp.thetas, resp.vs)
    return resp
end

function run_controller()
    veh = Vehicle(2.0, 2.0, pi/4, 0.0)
    @info "Client loop starting" cycles=N_CYCLES
    planning_rate = Rate(1/loop_time)
     #Time convertion to Hz
    inner_loop_counter = loop_time/DT
    global_idx = 0
    t_now = time()

    for k in 1:N_CYCLES 
        t0 = time() # Has to be the first
        t_rel = round(time() - t_now; digits = 3)
        action_dict[t_rel] = (ACTION_STEERING, ACTION_SPEED)
        # global_idx += 1  # dict indexing counter
        resp = predict_vehicle_state(veh, t_rel)
        # println(resp) <- Take lots of time to print. Not recommended if you need timestamps
        for i in eachindex(resp.xs)
            # global_idx += 1
            veh = Vehicle(resp.xs[i], resp.ys[i], resp.thetas[i], resp.vs[i])
            veh_traj_dict[resp.ts[i]] = veh
        end
        # @info "cycle $k → vehicle" veh
        # integ_time = time() - t0
        # @info "Doing maths for $integ_time"
        # @info "going to sleep"
        sleep(planning_rate)
        # sleeptime = time() - t0
        # @info "Loop duration is $sleeptime"
    end
    @info "Finished $N_CYCLES cycles"
    # @info "veh_traj_dict" veh_traj_dict
    # @info "action_dict"   action_dict
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
using Serialization
serialize("veh_log.jls", veh_traj_dict)     
@info "Wrote $(length(veh_traj_dict)) poses to veh_log.jls"