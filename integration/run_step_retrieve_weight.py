from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, pinch_finger

set_env('long')
runner_init()

reset_reward()
# pinch_finger("microwave_handle")
minimize_l2_distance_reward("palm", "weight")
minimize_l2_distance_reward("weight", "weightsensor", primary_reward=True)
execute_plan()
