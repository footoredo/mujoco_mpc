import os
import grpc
from PIL import Image
import mujoco
import mujoco.viewer
import mujoco_viewer
import cv2
import ffmpeg
import mujoco_mpc
# print(mujoco_mpc.__file__)
from mujoco_mpc import agent as agent_lib
import numpy as np
import joblib


import pathlib


def vidwrite(fn, images, framerate=60, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


def get_observation(model, data):
  del model
  return np.concatenate([data.qpos, data.qvel])


def environment_step(model, data, action):
  data.ctrl[:] = action
  mujoco.mj_step(model, data)
  return get_observation(model, data)


def environment_reset(model, data):
  mujoco.mj_resetData(model, data)
  return get_observation(model, data)


ENV = "cabinet"
# ENV = "kitchen"
SAVE_VIDEO = True

REWARD_CNT = {
    "min_l2": 0,
    "max_l2": 0,
    "joint": 0,
}

TASK_PARAMS = {}
COST_WEIGHTS = {}

CABINET_NAME_MAPPING = {
    "palm": "hand",
    "red_block": "box",
    "red_block_left_side": "box_left",
    "red_block_right_side": "box_right",
    "left_wooden_cabinet_handle": "leftdoorhandle",
    "right_wooden_cabinet_handle": "rightdoorhandle",
    "left_wooden_cabinet": "leftdoorhinge",
    "right_wooden_cabinet": "rightdoorhinge",
    "yellow_cube": "target",
    "right_wooden_cabinet_inside": "target_position"
}

KITCHEN_NAME_MAPPING = {
    "palm": "hand",
    "left_cabinet_handle": "cabinet_doorhandle_l",
    "right_cabinet_handle": "cabinet_doorhandle_r",
    "left_cabinet": "leftdoorhinge",
    "right_cabinet": "rightdoorhinge",
    "microwave_handle": "microwave_handle",
    "microwave": "micro0joint",
    "blue_kettle_handle": "kettle_handle"
}

NAME_MAPPING = {
    "cabinet": CABINET_NAME_MAPPING,
    "kitchen": KITCHEN_NAME_MAPPING
}

PRIMARY_REWARD = None

COST_NAMES_REQUIRED = []


def reset_reward():
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    for key in REWARD_CNT.keys():
        REWARD_CNT[key] = 0
    TASK_PARAMS = {}
    COST_WEIGHTS = {}
    PRIMARY_REWARD = None
    COST_NAMES_REQUIRED = []


def map_name(name):
    return NAME_MAPPING[ENV][name]


def minimize_l2_distance_reward(obj1, obj2, primary_reward=False):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    REWARD_CNT["min_l2"] += 1
    cnt = REWARD_CNT["min_l2"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"Reach{cnt}ObjectA"] = map_name(obj1)
    TASK_PARAMS[f"Reach{cnt}ObjectB"] = map_name(obj2)
    COST_WEIGHTS[f"Reach{cnt}"] = 1.0

    if primary_reward:
        PRIMARY_REWARD = f"Reach{cnt}"

    COST_NAMES_REQUIRED.append(f"Reach{cnt}")


def maximize_l2_distance_reward(obj1, obj2, distance=0.5, primary_reward=False):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    REWARD_CNT["max_l2"] += 1
    cnt = REWARD_CNT["max_l2"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"MoveAway{cnt}ObjectA"] = map_name(obj1)
    TASK_PARAMS[f"MoveAway{cnt}ObjectB"] = map_name(obj2)
    TASK_PARAMS[f"MoveAwayDistance"] = distance * 1.2
    COST_WEIGHTS[f"Move Away{cnt}"] = 1.0

    if primary_reward:
        PRIMARY_REWARD = f"Move Away{cnt}"

    COST_NAMES_REQUIRED.append(f"Move Away{cnt}")


def set_joint_fraction_reward(obj, fraction, primary_reward=False):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    REWARD_CNT["joint"] += 1
    cnt = REWARD_CNT["joint"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"JointTarget{cnt}"] = map_name(obj)
    TASK_PARAMS[f"JointTarget{cnt}Angle"] = fraction * 1.5
    COST_WEIGHTS[f"Joint Target{cnt}"] = 1.0

    if primary_reward:
        PRIMARY_REWARD = f"Joint Target{cnt}"

    COST_NAMES_REQUIRED.append(f"Joint Target{cnt}")


def set_primary_reward(reward_index):
    global PRIMARY_REWARD
    PRIMARY_REWARD = COST_NAMES_REQUIRED[reward_index]


last_primary = None

def _execute(task="kitchen", custom=False, use_viewer=True, init_data=None, save_video=True, save_last_img=False, verbose=False):
    # ctx = mujoco.GLContext(1920, 1080)
    # ctx.make_current()

    model_path = (
        pathlib.Path(__file__).parent.parent
        / f"build/mjpc/tasks/panda/{task}/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if init_data is None:
        data = mujoco.MjData(model)
    else:
        data = init_data

    mj_viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen', width=1280, height=960)
    # mj_viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen', width=640, height=480)

    # viewer = mujoco_viewer.MujocoViewer(model, data)

    # img = viewer.read_pixels(camid=2)
    # img = Image.fromarray(img)
    # img.save("initial.png")

    # viewer.render()

    # input()

    repeats = 5
    control_timestep = model.opt.timestep * repeats

    actions = []
    observations = []

    images = []

    # "<camera pos="1.353 -0.887 1.285" xyaxes="0.556 0.831 0.000 -0.559 0.374 0.740"/>"

    with agent_lib.Agent(task_id=f"Panda {task.capitalize()}", model=model) as agent:
        def plan(step_limit=500, cost_limit=None, cost_name=None, viewer=None):
            global last_primary
            if cost_name.startswith("Default Pose"):
                last_primary = None
            else:
                last_primary = cost_name
            for i in range(step_limit):
                agent.set_state(
                    time=data.time,
                    qpos=data.qpos,
                    qvel=data.qvel,
                    act=data.act,
                    mocap_pos=data.mocap_pos,
                    mocap_quat=data.mocap_quat,
                    userdata=data.userdata,
                )
                agent.planner_step()
                # print(i)
                # actions.append(agent.get_action(averaging_duration=control_timestep))
                actions.append(agent.get_action())
                # print(i)
                for _ in range(repeats):
                    observations.append(environment_step(model, data, actions[-1]))
                    if viewer is not None:
                        viewer.sync()
                if save_video:
                    img = mj_viewer.read_pixels(camid=0)
                    images.append(img)
                    # total_cost += agent.get_total_cost()
                if cost_name is None:
                    cost = agent.get_total_cost()
                else:
                    cost = agent.get_cost_term_values()[cost_name]
                if i % 20 == 0 and verbose:
                    # print(i, cost)
                    print(i, f"{agent.get_total_cost():.2f} {cost:.2f}", agent.get_cost_term_values())
                # agent.planner_step()
                if cost_limit is not None and cost <= cost_limit:
                    return True
                # observations.append(environment_step(model, data, actions[-1]))
                # viewer.render()
            return False
        
        def run_once(task_parameters, cost_weights, cost_limit, cost_name=None, step_limit=1000, viewer=None):
            if task_parameters is not None:
                agent.set_task_parameters(task_parameters)
            cost_names = agent.get_cost_term_values().keys()
            zeroed_cost_weights = {
                key: cost_weights.get(key, 0.0) for key in cost_names
            }
            agent.set_cost_weights(zeroed_cost_weights)
            return plan(cost_limit=cost_limit, cost_name=cost_name, step_limit=step_limit, viewer=viewer)
        
        def run_reset(viewer=None):
            print(last_primary)
            succ = run_once(task_parameters=None, cost_weights={
                "Default Pose": 1, last_primary: 1
            }, cost_limit=0.02, cost_name="Default Pose", viewer=viewer, step_limit=200)
            return succ

        def run_reset_no_obstruction(viewer=None):
            succ = run_once(task_parameters=None, cost_weights={
                "Default Pose No-Obstruction": 1, last_primary: 1
            }, cost_limit=0.02, cost_name="Default Pose No-Obstruction", viewer=viewer, step_limit=200)
            return succ

        def run_with_retries(task_name, task_parameters, cost_weights, cost_limit, cost_name=None, num_retries=3, step_limit=1000, viewer=None):
            for i in range(num_retries):
                print(f"Task [{task_name}] retry #{i} ...")
                run_reset(viewer=viewer)
                succ = run_once(task_parameters, cost_weights, cost_limit, cost_name=cost_name, step_limit=step_limit, viewer=viewer)
                costs = agent.get_cost_term_values()
                for name in sorted(cost_weights.keys()):
                    print(name, costs[name])
                print()
                if succ:
                    return True
            return False
        
        def run_kitchen(viewer=None):
            succ = run_with_retries("move kettle", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "kettle_handle",
                "MoveAwayObjectA": "kettle_center",
                "MoveAwayObjectB": "microwave_handle",
                "MoveAwayDistance": 0.7
            }, cost_weights={"Reach": 1.0, "Move Away": 1.0},
            cost_limit=0.02, cost_name="Move Away", num_retries=3, viewer=viewer)

            if not succ:
                return False

            succ = run_with_retries("open microwave", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "microwave_handle",
                "JointTarget": "micro0joint",
                "JointTargetAngle": 1.2
            }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
            cost_limit=0.02, cost_name="Joint Target", num_retries=3, viewer=viewer)

            if not succ:
                return False

            succ = run_with_retries("move cube", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "box",
                "Reach2ObjectA": "box",
                "Reach2ObjectB": "target_position"
            }, cost_weights={"Reach": 1.0, "Reach2": 1.0},
            cost_limit=0.02, cost_name="Reach2", num_retries=3, viewer=viewer)

            return succ
        
        def run_kitchen_cabinet_l(viewer=None):
            succ = run_with_retries("open cabinet_l", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "cabinet_doorhandle_l",
                "JointTarget": "leftdoorhinge",
                "JointTargetAngle": 1.5
            }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
            cost_limit=0.02, cost_name="Joint Target", num_retries=3, viewer=viewer)

            return succ

        def run_kitchen_cabinet_r(viewer=None):
            succ = run_with_retries("open cabinet_r", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "cabinet_doorhandle_r",
                "JointTarget": "rightdoorhinge",
                "JointTargetAngle": 1.5
            }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
            cost_limit=0.02, cost_name="Joint Target", num_retries=3, viewer=viewer)

            return succ
        
        def run_kitchen_cabinet_both(viewer=None):
            succ = run_kitchen_cabinet_r(viewer)
            if not succ:
                return False
            succ = run_kitchen_cabinet_l(viewer)
            run_reset_no_obstruction(viewer=viewer)
            return succ

        def run_kitchen_dummy(viewer=None):
            succ = run_reset(viewer)
            return succ

        def run_cabinet(viewer=None):
            succ = run_with_retries("remove stick", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "box_right",
                "MoveAwayObjectA": "box_right",
                "MoveAwayObjectB": "rightdoorhandle",
                "MoveAwayDistance": 0.6
            }, cost_weights={"Reach": 1.0, "Move Away": 1.0},
            cost_limit=0.02, cost_name="Move Away", num_retries=3, viewer=viewer)

            if not succ:
                return False
            
            succ = run_with_retries("open door", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "rightdoorhandle",
                "JointTarget": "rightdoorhinge",
                "JointTargetAngle": 1.5
            }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
            cost_limit=0.02, cost_name="Joint Target", num_retries=3, viewer=viewer)

            if not succ:
                return False
            
            succ = run_with_retries("move cube", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "target",
                "Reach2ObjectA": "target",
                "Reach2ObjectB": "target_position"
            }, cost_weights={"Reach": 1.0, "Reach2": 1.0},
            cost_limit=0.02, cost_name="Reach2", num_retries=3, viewer=viewer)

            return succ
        
        def run_cabinet_test1(viewer=None):
            succ = run_with_retries("remove stick", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "box_right",
                "MoveAwayObjectA": "box_right",
                "MoveAwayObjectB": "rightdoorhandle",
                "MoveAwayDistance": 0.6
            }, cost_weights={"Reach": 1.0, "Move Away": 1.0},
            cost_limit=0.02, cost_name="Move Away", num_retries=3, viewer=viewer)

            return succ
        
        def run_cabinet_test2(viewer=None):
            succ = run_with_retries("open door", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "rightdoorhandle",
                "JointTarget": "rightdoorhinge",
                "JointTargetAngle": 1.5
            }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
            cost_limit=0.02, cost_name="Joint Target", num_retries=3, viewer=viewer)

            return succ

        def run_cabinet_dummy(viewer=None):
            succ = run_with_retries("remove stick", task_parameters={
                "ReachObjectA": "rightdoorhandle",
                "ReachObjectB": "leftdoorhandle",
            }, cost_weights={"Reach": 1.0},
            cost_limit=0.02, cost_name="Reach", num_retries=1, step_limit=50, viewer=viewer)
            input()

            return succ

        def run_custom(viewer=None):
            if PRIMARY_REWARD is None:
                cost_limit = 0.01 * sum(list(COST_WEIGHTS.values()))
            else:
                cost_limit = 0.02
            succ = run_with_retries("custom", task_parameters=TASK_PARAMS, cost_weights=COST_WEIGHTS,
            cost_limit=cost_limit, cost_name=PRIMARY_REWARD, num_retries=3, viewer=viewer)

            run_reset_no_obstruction(viewer)

            return succ

        if custom:
            run_fn = run_custom
        else:
            if task == "kitchen":
                run_fn = run_kitchen_cabinet_both
                # run_fn = run_kitchen_dummy
            elif task == "cabinet":
                run_fn = run_cabinet_dummy
            else:
                raise NotImplementedError()

        if use_viewer:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                ret = run_fn(viewer=viewer)
                input()
        else:
            ret = run_fn(viewer=None)

        if save_video:
            images = np.stack(images, 0)
            vidwrite("output.mp4", images, 240)

        if save_last_img:
            im = mj_viewer.read_pixels(camid=0)
            cv2.imwrite('output.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        joblib.dump(data, "data.joblib")

        with open("outcome", "w") as outcome_f:
            outcome_f.write(str(int(ret)))

        return ret


def execute_plan(duration=None):
    try:
        init_data = joblib.load("data.joblib")
    except FileNotFoundError:
        init_data = None

    print(TASK_PARAMS)
    print(COST_WEIGHTS)
    print(PRIMARY_REWARD)
    
    print(int(_execute(ENV, custom=True, use_viewer=False, save_video=SAVE_VIDEO, save_last_img=True, init_data=init_data)))


if __name__ == "__main__":

    init_data = None

    # try:
    #     init_data = joblib.load("data.joblib")
    # except FileNotFoundError:
    #     init_data = None

    T = 1
    n_succ = 0
    for _ in range(T):
        n_succ += int(_execute("kitchen", use_viewer=True, save_video=False, save_last_img=True, init_data=init_data))
    
    print(n_succ)
