from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('long')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "hinge_cabinet_door_handle")
# maximize_l2_distance_reward("hinge_cabinet_door_handle", "weight_sensor_lock")
set_joint_fraction_reward("hinge_cabinet", 1, primary_reward=True)
execute_plan(reset_after_done=False)