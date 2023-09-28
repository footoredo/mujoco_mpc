from core import set_env, runner_init, end_effector_to, get_object_position, finish

set_env("cabinet")
runner_init()

cube_pos = get_object_position("yellow_cube")
print(cube_pos)
end_effector_to(cube_pos)

finish()
