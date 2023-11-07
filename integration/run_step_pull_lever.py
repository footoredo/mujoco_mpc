from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, stack_reward, lift, \
    set_joint_fraction_reward, execute_plan, set_env, runner_init, set_repeats, get_joint_value, set_retries
    

set_retries(1)

set_repeats(5)
set_env('locklock')
runner_init()

# reset_reward()
# print("joint", get_joint_value("red_lever_joint"))
# minimize_l2_distance_reward("palm", "red_block")
# set_joint_fraction_reward("red_lever_joint", -1, primary_reward=True)
# execute_plan()
# print("joint", get_joint_value("red_lever_joint"))


# reset_reward()
# minimize_l2_distance_reward("palm", "rightdoorhandle")
# set_joint_fraction_reward("rightdoorhinge", 1, primary_reward=True)
# execute_plan(finish=False, reset_after_done=False)


reset_reward()
minimize_l2_distance_reward("palm", "red_switch_handle")
set_joint_fraction_reward("red_switch_handle_joint", 1, primary_reward=True)
# execute_plan(finish=False, reset_after_done=False)


# reset_reward()
# minimize_l2_distance_reward("palm", "rightdoorhandle")
# set_joint_fraction_reward("rightdoorhinge", 1, primary_reward=True)
execute_plan()