from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('kitchen')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "microwave_handle")
set_joint_fraction_reward("microwave", 1.0, primary_reward=True)
execute_plan()