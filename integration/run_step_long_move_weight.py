from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('long')
runner_init()

reset_reward()
minimize_l2_distance_reward("palm", "weight")
maximize_l2_distance_reward("hinge_cabinet_inside", "weight")
execute_plan(reset_after_done=False)