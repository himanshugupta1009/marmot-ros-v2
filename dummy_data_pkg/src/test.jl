#!/usr/bin/env julia
using RobotOS

# 1  Import the standard service type
@rosimport std_srvs.srv: SetBool

# 2  Generate / load the Julia wrappers
rostypegen()                             # ← this is the only generator call
using .std_srvs.srv

# 3  Service callback (no type annotations needed)
function handle(req)                           #WORKS PERFECTLY
    @info "Received SetBool request"
    println("Request data: ", req.data)
    return SetBoolResponse(true, "Confirmed")
end

function main()
    RobotOS.init_node("test_service_node")

    # Advertise the service
    RobotOS.Service("/setbool", SetBool, handle)
    @info "Service /setbool advertised"

    # Enter the event loop
    RobotOS.spin()
end

main()
