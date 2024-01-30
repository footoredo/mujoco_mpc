from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, stack_reward, lift, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, set_repeats

set_repeats(2)
set_env('blocks')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "red_block")
lift("red_block", height=0.12)
minimize_l2_distance_reward("red_block", "red_bin", primary_reward=True)
execute_plan(reset_after_done=False)
