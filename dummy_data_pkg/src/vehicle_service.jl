#!/usr/bin/env julia

# vehicle_service.jl - Service provider only
using RobotOS
@rosimport dummy_data_pkg.msg: PeoplePoseArray
@rosimport dummy_data_pkg.srv: MoveVehicle
rostypegen()
using .dummy_data_pkg.msg: PeoplePoseArray
using .dummy_data_pkg.srv: MoveVehicle, MoveVehicleRequest, MoveVehicleResponse

include(joinpath(@__DIR__, "struct_definition.jl"))
include(joinpath(@__DIR__, "utils.jl"))

const VEHICLE_L = 0.75
const SUB_STEPS = 5

function handle(req::MoveVehicleRequest)#::MoveVehicleResponse
    veh = Vehicle(req.x, req.y, req.theta, req.v)
    xs = Float64[];  ys = Float64[];  thetas = Float64[];  vs = Float64[];   ts = Float64[]

    for j in 1:SUB_STEPS 
        nx, ny, nθ = move_vehicle(
            veh.x, veh.y, veh.theta,
            VEHICLE_L,
            req.steering,            # constant steering (rad)
            req.speed,               # constant speed   (m/s)
            req.dt                   # integration timestep for the substep(s)
        )
        push!(xs, nx);  push!(ys, ny);  push!(thetas, nθ);  push!(vs, req.speed)
        push!(ts, req.t0 + j * req.dt)
        veh = Vehicle(nx, ny, nθ, req.speed)
    end

    resp = MoveVehicleResponse()
    resp.xs     = xs
    resp.ys     = ys
    resp.vs     = vs
    resp.thetas = thetas
    resp.ts     = ts
    return resp                       
end

function main()
    RobotOS.init_node("vehicle_server"; anonymous=true)
    Service("/move_vehicle", MoveVehicle, handle)
    @info " /move_vehicle service ready"
    RobotOS.spin()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end