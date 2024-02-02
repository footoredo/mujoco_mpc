from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, pinch_finger

set_env('cabinet')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "box_right")
maximize_l2_distance_reward("box_right", "rightdoorhandle", primary_reward=True)
execute_plan()
