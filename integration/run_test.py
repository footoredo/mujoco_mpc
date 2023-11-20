from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('blocks')
runner_init(load_init=False)

reset_reward()
minimize_l2_distance_reward("palm", "crate")
maximize_l2_distance_reward("red_block", "crate", primary_reward=True)
execute_plan()
