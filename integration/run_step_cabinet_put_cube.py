from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, pinch_finger

set_env('cabinet')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "yellow_cube")
minimize_l2_distance_reward("yellow_cube", "target_position_in_wooden_cabinet", primary_reward=True)
execute_plan()
