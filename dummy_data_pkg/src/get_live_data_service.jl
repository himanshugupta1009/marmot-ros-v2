#!/usr/bin/env julia
# =============================================================================
# File: get_live_data_service.jl
# Purpose:
#   Expose a ROS service that returns the most recent LiveData snapshot
#   published on /car/sim/LiveData by the realtime vehicle node.
#
# ROS Interface
#   Subscribes: /car/sim/LiveData     (dummy_data_pkg/LiveData)
#   Service   : /car/sim/GetLiveData  (dummy_data_pkg/GetLiveData)
#   Request   : GetLiveDataRequest()  # empty
#   Response  : GetLiveDataResponse(snaps::Vector{LiveData})
#               — this implementation returns a 1-element vector.
#
# Data Flow & Timing
#   • A subscriber callback caches the latest LiveData into LATEST (Ref cell).
#   • Each service call returns that cached value as a single-element vector,
#     then clears LATEST to a default-constructed message.
#
# Key Variables
#   • LATEST::Ref{LiveData} — single-item cache updated by subscriber and read
#     by the service handler. Acts like “most recent snapshot”.
#
# Assumptions
#   • Another node (e.g., realtime_vehicle_node.jl) publishes LiveData regularly.
#   • The client tolerates empty/default LiveData if called before the first
#     message arrives or if LATEST has been cleared between calls.
#
# Gotchas
#   • Clearing behavior: after serving, LATEST is reset to LiveData(). A second
#     immediate call will likely return an “empty” message unless a new one has
#     arrived. Remove the reset if you prefer “sticky” last value semantics.
#   • Subscriber creation before init_node(): this file constructs the subscriber
#     at top-level. The canonical pattern is to call RobotOS.init_node() first
#     (e.g., inside main) and then create Subscriber; leaving it as-is preserves
#     current behavior.
#   • Concurrency: the callback may update while the service reads. Using Ref
#     is adequate for a “latest-or-previous” snapshot; no deep copy is done.
# =============================================================================

using RobotOS
@rosimport dummy_data_pkg.msg: LiveData
@rosimport dummy_data_pkg.srv: GetLiveData
rostypegen()
using .dummy_data_pkg.msg: LiveData
using .dummy_data_pkg.srv: GetLiveData, GetLiveDataRequest, GetLiveDataResponse

# Optionally brings in types used elsewhere (kept for compatibility).
include(joinpath(@__DIR__, "struct_definition.jl"))

# Single-element cache of the most recent LiveData published.
const LATEST = Ref(LiveData())  # default-construct an empty message

"""
live_cb(msg::LiveData)

Subscriber callback. Overwrites the cached snapshot with the newly received
LiveData from /car/sim/LiveData.

Args:
  msg::LiveData — the incoming telemetry snapshot.
Returns:
  nothing
"""
function live_cb(msg::LiveData)
    LATEST[] = msg
end

# Note: constructed at top-level to preserve existing behavior.
Subscriber("/car/sim/LiveData", LiveData, live_cb)

"""
handle(req::GetLiveDataRequest) -> GetLiveDataResponse

Service handler that returns the most recent LiveData snapshot as a
single-element vector `snaps`. After responding, resets the cache to a
default-constructed LiveData().

Args:
  req::GetLiveDataRequest — unused (empty request type).
Returns:
  GetLiveDataResponse — with `snaps = [LATEST[]]`.
"""
function handle(req::GetLiveDataRequest)::GetLiveDataResponse
    resp = GetLiveDataResponse()
    resp.snaps = [LATEST[]]   # return a 1-element vector
    println("live_data_received")
    LATEST[] = LiveData()     # clear cache; remove if you want sticky behavior
    return resp
end

"""
main()

Initialize the node and advertise the /car/sim/GetLiveData service.
Blocks via RobotOS.spin().
"""
function main()
    RobotOS.init_node("get_live_data_service")
    srv = Service("/car/sim/GetLiveData", GetLiveData, handle)
    println("[/car/sim/GetLiveData] Node ready, waiting for incoming requests…")
    RobotOS.spin()
end

# Allow running as a script or importing without side effects.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
