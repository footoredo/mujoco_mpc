from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('long')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "green_weight_sensor_lock")
minimize_l2_distance_reward("wooden_cabinet_door_handle", "green_weight_sensor_lock", primary_reward=True)
execute_plan()