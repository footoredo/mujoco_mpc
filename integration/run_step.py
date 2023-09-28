from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan

reset_reward()
minimize_l2_distance_reward("palm", "yellow_cube")
minimize_l2_distance_reward("yellow_cube", "right_wooden_cabinet_inside", primary_reward=True)
set_joint_fraction_reward("right_wooden_cabinet", 1)
execute_plan()
