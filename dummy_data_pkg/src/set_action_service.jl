#!/usr/bin/env julia

using RobotOS
@rosimport dummy_data_pkg.srv: SetAction
rostypegen()
using .dummy_data_pkg.srv: SetAction, SetActionRequest, SetActionResponse

# include(joinpath(@__DIR__, "realtime_vehicle_node.jl"))  # for ACTION_ARRAY
include(joinpath(@__DIR__, "struct_definition.jl"))

function handle(req::SetActionRequest)::SetActionResponse
    # Round up to next 0.5-second step
    t = ceil(req.t / 0.5) * 0.5
    ACTION_ARRAY[t] = (req.steering, req.speed)
    msg = "Stored action at t=$t → (steering=$(req.steering), speed=$(req.speed))"
    @info msg
    return SetActionResponse(true, msg)
end

function main()
    RobotOS.init_node("set_action_service")
    RobotOS.Service("/set_action", SetAction, handle)
    println("[set_action_service] Node ready, waiting for incoming requests…")
    RobotOS.spin()
end

main()

