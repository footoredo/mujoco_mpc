from core import reset_reward, set_min_l2_distance_reward, set_max_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan


reset_reward()

set_min_l2_distance_reward("hand", "box_right")
set_max_l2_distance_reward("rightdoorhandle", "box_right")

execute_plan()