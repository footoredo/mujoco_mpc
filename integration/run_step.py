from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('blocks')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "red_cube")
minimize_l2_distance_reward("red_cube", "crate")
maximize_l2_distance_reward("crate", "yellow_cube", distance=0.5)
execute_plan()