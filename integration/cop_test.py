from core import set_env, cop_runner_init, end_effector_to, end_effector_open, end_effector_close, get_object_position, finish

set_env("cabinet")
cop_runner_init()

cube_pos = get_object_position("yellow_cube")
# print(cube_pos)
end_effector_open()
end_effector_to(cube_pos)
end_effector_close()

finish()
