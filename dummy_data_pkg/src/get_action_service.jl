#!/usr/bin/env julia
# =============================================================================
# File: get_action_service.jl
# Purpose:
#   ROS service node that returns the scheduled control (steering, speed)
#   for a given timestamp key.
#
# ROS Interface:
#   Service  : /get_action    (dummy_data_pkg/GetAction)
#   Request  : GetActionRequest(time::Float64)     # query key (seconds)
#   Response : GetActionResponse(steer::Float64,
#                                speed::Float64,
#                                found::Bool)      # true if key existed
#
# Timing & Keys:
#   • ACTION_ARRAY uses the exact Float64 request time as key (no tolerance).
#     The typical pattern is to index by the action-block start time (t_start).
#
# Key Variables:
#   • ACTION_ARRAY::Dict{Float64, Tuple{Float64,Float64}}
#       Maps time [s] → (steering [rad], speed [m/s]).
#
# Assumptions:
#   • Another handler (e.g., set_action_service.jl) inserts into ACTION_ARRAY.
#
# Gotchas:
#   • Exact float key match: if the client’s time differs by milliseconds,
#     the lookup will fail. Consider using a tolerance/nearest-key strategy
#     in the client or normalizing times when populating ACTION_ARRAY.
#   • Cross-process memory: ACTION_ARRAY is in this process only. If your
#     set_action service runs in a different process, they will NOT share
#     this Dict. To share, run both handlers in the same process, write to
#     a common store (e.g., Parameter Server/topic), or merge services.
# =============================================================================

using RobotOS
@rosimport dummy_data_pkg.srv: GetAction
rostypegen()
using .dummy_data_pkg.srv: GetAction, GetActionRequest, GetActionResponse

# The per-node action table:
const ACTION_ARRAY = Dict{Float64,Tuple{Float64,Float64}}()

"""
handle_get_action(req::GetActionRequest) -> GetActionResponse

Lookup the requested time key in ACTION_ARRAY and return (steer, speed).
Returns `found=false` with zeros if the key is absent.

Args:
  req.time::Float64  # seconds (must match an existing key exactly)

Response:
  GetActionResponse(steer::Float64, speed::Float64, found::Bool)
"""
function handle_get_action(req::GetActionRequest)::GetActionResponse
    println("REQUEST RECEIVED")
    t = req.time
    if haskey(ACTION_ARRAY, t)
        s, v = ACTION_ARRAY[t]
        return GetActionResponse(s, v, true)
    else
        return GetActionResponse(0.0, 0.0, false)
    end
end

"""
main()

Initialize the node and advertise the /get_action service.
Blocks forever via RobotOS.spin().
"""
function main()
    RobotOS.init_node("get_action_service")
    RobotOS.Service("/get_action", GetAction, handle_get_action)
    println("[get_action_service] Node ready, waiting for incoming requests…")
    RobotOS.spin()
end

main()
