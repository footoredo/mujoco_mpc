from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, set_repeats

set_repeats(5)
set_env('blocks')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "apple")
minimize_l2_distance_reward("apple", "palm")
execute_plan()