import os
import shutil
import pathlib
import subprocess


RUN_NAME = "test-newdefault"
# ENV = "cabinet"
ENV = "kitchen"

LLM_WORKDIR = pathlib.Path("/home/footoredo/playground/adaptivereplanning")
MPC_WORKDIR = pathlib.Path("/home/footoredo/playground/mujoco_mpc/integration")

MPC_RUN_HEADER = f"""from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \\
    set_joint_fraction_reward, execute_plan, set_env, runner_init

set_env('{ENV}')
runner_init()

"""


def clean_mpc():
    try:
        os.remove(MPC_WORKDIR / "data.joblib")
    except FileNotFoundError:
        pass


def run_mpc(step, reward_code):
    code_filename = MPC_WORKDIR / "run_step.py"
    log_filename = MPC_WORKDIR / "output"

    with open(code_filename, "w") as code_f:
        code_f.write(MPC_RUN_HEADER)
        code_f.write(reward_code)

    with open(log_filename, "w") as log_f:
        subprocess.check_call(["python", code_filename], cwd=MPC_WORKDIR, stdout=log_f)

    logdir = MPC_WORKDIR / "logs" / ENV / RUN_NAME / f"step-{step}"
    os.makedirs(logdir, exist_ok=True)

    image_filename = MPC_WORKDIR / "output.png"
    video_filename = MPC_WORKDIR / "output.mp4"
    data_filename = MPC_WORKDIR / "data.joblib"
    outcome_filename = MPC_WORKDIR / "outcome"

    with open(outcome_filename, "r") as outcome_f:
        outcome = int(outcome_f.readline().strip())

    for fn in [code_filename, log_filename, image_filename, video_filename, data_filename, outcome_filename]:
        try:
            shutil.copyfile(fn, logdir / fn.name)
        except FileNotFoundError:
            print(f"[{fn.name}] not found. skip copying.")

    shutil.copyfile(image_filename, LLM_WORKDIR / "run_images" / "step.png")

    return outcome


if __name__ == "__main__":
    reward_code_tests_kitchen = [
"""reset_reward()
minimize_l2_distance_reward("palm", "right_cabinet_handle")
set_joint_fraction_reward("right_cabinet", 1, primary_reward=True)

execute_plan(2)""", 
"""reset_reward()
minimize_l2_distance_reward("palm", "left_cabinet_handle")
set_joint_fraction_reward("left_cabinet", 1, primary_reward=True)

execute_plan()""",
"""reset_reward()
minimize_l2_distance_reward("palm", "microwave_handle")
set_joint_fraction_reward("microwave", 1, primary_reward=True)

execute_plan(2)""",
"""reset_reward()
minimize_l2_distance_reward("palm", "blue_kettle_handle")
maximize_l2_distance_reward("blue_kettle_handle", "microwave_handle", primary_reward=True)
execute_plan(2)
""", """reset_reward()
minimize_l2_distance_reward("palm", "microwave_handle")
set_joint_fraction_reward("microwave", 1, primary_reward=True)

execute_plan(2)""",
"""reset_reward()
minimize_l2_distance_reward("palm", "cube")
minimize_l2_distance_reward("cube", "target_position", primary_reward=True)

execute_plan(2)"""]

    reward_code_tests_cabinet = [
"""reset_reward()
minimize_l2_distance_reward("palm", "yellow_cube", primary_reward=True)
minimize_l2_distance_reward("yellow_cube", "yellow_cube")
execute_plan(2)
""",
"""reset_reward()
minimize_l2_distance_reward("palm", "right_wooden_cabinet_handle")
set_joint_fraction_reward("right_wooden_cabinet", 1, primary_reward=True)

execute_plan(2)
""",
"""reset_reward()
minimize_l2_distance_reward("palm", "yellow_cube", primary_reward=True)
execute_plan(2)
""",
"""reset_reward()
minimize_l2_distance_reward("palm", "red_block_right_side")
maximize_l2_distance_reward("red_block_right_side", "right_wooden_cabinet_handle", primary_reward=True)
execute_plan()
""",
"""reset_reward()
minimize_l2_distance_reward("palm", "right_wooden_cabinet_handle")
set_joint_fraction_reward("right_wooden_cabinet", 1, primary_reward=True)

execute_plan(2)
""",
"""reset_reward()
minimize_l2_distance_reward("palm", "yellow_cube")
minimize_l2_distance_reward("yellow_cube", "right_wooden_cabinet_inside", primary_reward=True)
set_joint_fraction_reward("right_wooden_cabinet", 1)
execute_plan()
"""
    ]

    reward_code_tests_cabinet_new = [
"""reset_reward()
minimize_l2_distance_reward("palm", "yellow_cube")
execute_plan()""",
"""reset_reward()
minimize_l2_distance_reward("palm", "right_wooden_cabinet_handle")
set_joint_fraction_reward("right_wooden_cabinet", 1.0, primary_reward=True)
execute_plan()""",
"""reset_reward()
minimize_l2_distance_reward("palm", "red_block_right_side")
execute_plan()
""",
"""reset_reward()
minimize_l2_distance_reward("palm", "red_block_right_side")
maximize_l2_distance_reward("red_block_right_side", "right_wooden_cabinet_handle", primary_reward=True)
execute_plan()""",
"""reset_reward()
minimize_l2_distance_reward("palm", "right_wooden_cabinet_handle")
set_joint_fraction_reward("right_wooden_cabinet", 1.0, primary_reward=True)
execute_plan()""",
"""reset_reward()
minimize_l2_distance_reward("palm", "yellow_cube")
minimize_l2_distance_reward("yellow_cube", "right_wooden_cabinet_inside", primary_reward=True)
set_joint_fraction_reward("right_wooden_cabinet", 1.0)
execute_plan()"""
    ]

    # reward_code_list = reward_code_tests_cabinet_new
    reward_code_list = reward_code_tests_kitchen

    clean_mpc()

    for i in range(0, len(reward_code_list)):
        print(f"Running step-{i + 1} ...")
        outcome = run_mpc(i + 1, reward_code_list[i])
        print(f"Step-{i + 1} outcome:", ["fail", "success"][outcome])
