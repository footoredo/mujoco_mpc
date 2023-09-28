from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('cabinet')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "wooden_cabinet_handle")
set_joint_fraction_reward("wooden_cabinet", 1.0, primary_reward=True)
execute_plan()