import os
import shutil
import pathlib
import subprocess


RUN_NAME = "test1"

LLM_WORKDIR = pathlib.Path("/home/footoredo/playground/RePlan")
MPC_WORKDIR = pathlib.Path("/home/footoredo/playground/mujoco_mpc/integration")

MPC_RUN_HEADER = """from core import reset_reward, minimize_l2_distance_reward, maximize_l2_distance_reward, \\
    set_joint_fraction_reward, execute_plan

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

    logdir = MPC_WORKDIR / "logs" / RUN_NAME / f"step-{step}"
    os.makedirs(logdir, exist_ok=True)

    image_filename = MPC_WORKDIR / "output.png"
    data_filename = MPC_WORKDIR / "data.joblib"

    for fn in [code_filename, log_filename, image_filename, data_filename]:
        shutil.copyfile(fn, logdir / fn.name)


if __name__ == "__main__":

    reward_code_test = """reset_reward()
minimize_l2_distance_reward("palm", "left_cabinet_handle")
set_joint_fraction_reward("left_cabinet", 1.0, primary_reward=True)

execute_plan(2)"""

    clean_mpc()
    run_mpc(1, reward_code_test)
