std_msgs/Header header          # stamp = time of snapshot

float32   x
float32   y
float32   theta
float32   v

int32[]   ids                   # N humans
float32[] hx                    # N
float32[] hy                    # N
float32[] belief_pdf            # N * num_goals
