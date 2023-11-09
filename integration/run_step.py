from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('locklock')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "wooden_cabinet_door_handle")
minimize_l2_distance_reward("wooden_cabinet_door_handle", "wooden_cabinet_door")
set_joint_fraction_reward("wooden_cabinet_door", 1, primary_reward=True)
execute_plan()