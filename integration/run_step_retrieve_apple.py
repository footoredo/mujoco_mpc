from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, pinch_finger

set_env('kitchen')
runner_init()

reset_reward()
# pinch_finger("green_apple")
minimize_l2_distance_reward("palm", "green_apple")
minimize_l2_distance_reward("target_position", "green_apple", primary_reward=True)
execute_plan()