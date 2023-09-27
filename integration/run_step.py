from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan

reset_reward()
minimize_l2_distance_reward("palm", "left_cabinet_handle")
set_joint_fraction_reward("left_cabinet", 1.0, primary_reward=True)

execute_plan(2)