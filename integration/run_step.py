from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('kitchen')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "blue_kettle_handle", primary_reward=True)
maximize_l2_distance_reward("blue_kettle_handle", "cabinet_handle")
execute_plan()