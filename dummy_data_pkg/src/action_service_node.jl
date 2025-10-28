#!/usr/bin/env julia

using RobotOS

# Import service types
@rosimport dummy_data_pkg.srv: SetAction, GetAction
rostypegen()
using .dummy_data_pkg.srv: 
    SetAction, SetActionRequest, SetActionResponse,
    GetAction, GetActionRequest, GetActionResponse

# Shared action buffer
const ACTION_ARRAY = Dict{Float64,Tuple{Float64, Float64}}()

# Service callback: store actions
function handle_set_action(req::SetActionRequest)::SetActionResponse

    t = ceil(req.t / 0.5) * 0.5
    ACTION_ARRAY[t] = (req.steering, req.speed)
    msg = "Stored action at t=$t → (steering=$(req.steering), speed=$(req.speed))"
    @info msg
    return SetActionResponse(true, msg)

    # # req.time, req.steering, req.velocity are fields
    # ACTION_ARRAY[req.time] = (req.steering, req.velocity)
    # @info "[action_node] set_action: t=$(req.time) -> $(ACTION_ARRAY[req.time])"
    # return SetActionResponse(success=true)
end

# Service callback: retrieve actions
function handle_get_action(req::GetActionRequest)::GetActionResponse
    println("REQUEST RECEIVED")
    t = req.time
    if haskey(ACTION_ARRAY, t)
        s,v = ACTION_ARRAY[t]
        return GetActionResponse(s, v, true)
    else
        return GetActionResponse(0.0, 0.0, false)
    end
    # t = req.time
    # # lookup or default
    # s, v = get(ACTION_ARRAY, t, (0.0, 0.0))
    # found = haskey(ACTION_ARRAY, t)
    # @info "[action_node] get_action: t=$t -> (steering=$s, velocity=$v), found=$found"
    # return GetActionResponse(steering=s, velocity=v, found=found)
end

function main()
    # Initialize ROS node
    RobotOS.init_node("action_service_node")

    # Advertise both services on this node
    RobotOS.Service("/set_action",
                     SetAction,
                     handle_set_action)

    RobotOS.Service("/get_action",
                     GetAction,
                     handle_get_action)

    @info "[action_node] Services '/set_action' and '/get_action' advertised, spinning..."
    # Enter the service loop
    RobotOS.spin()

    # ACTION_ARRAY
        aa_keys = sort!(collect(keys(ACTION_ARRAY)))
        println("\nACTION_ARRAY keys → ", aa_keys)
        for t in aa_keys
            s, v = ACTION_ARRAY[t]
            println("t = $(round(t, digits=1)) (steering=$(round(s, digits=3)), velocity=$(round(v, digits=3)))")
        end
end

# If launched directly, run main()
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
