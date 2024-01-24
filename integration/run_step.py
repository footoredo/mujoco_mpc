from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('blocks')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "left_cube")
minimize_l2_distance_reward("left_cube", "crate", primary_reward=True)
execute_plan()