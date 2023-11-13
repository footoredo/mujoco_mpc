from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, stack_reward, lift, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, set_repeats

set_env('blocks')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "right_cube")
minimize_l2_distance_reward("right_cube", "crate", primary_reward=True)
execute_plan()
