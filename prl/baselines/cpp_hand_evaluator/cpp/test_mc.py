from hand_eval import *


# 1, 41, 18, 19, 16, 20, 24 must return 5586
# 51, 47, 43, 39, 35, 0, 1 must return a lower value than random num
print(run_monte_carlo(1, 41, 18, 19, 16, 20, 24, 1, 1000000))