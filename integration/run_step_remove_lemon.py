from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, stack_reward, lift, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, set_repeats

set_repeats(2)
set_env('blocks')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "yellow_block")
lift("yellow_block", height=0.10)
maximize_l2_distance_reward("yellow_block", "red_bin", primary_reward=True)
execute_plan(reset_after_done=False)