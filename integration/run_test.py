from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan

reset_reward()
minimize_l2_distance_reward("palm", "microwave_handle")
set_joint_fraction_reward("microwave", 1)

execute_plan(2)
