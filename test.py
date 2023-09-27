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


def test(task="kitchen", use_viewer=True):
    # ctx = mujoco.GLContext(1920, 1080)
    # ctx.make_current()

    model_path = (
        pathlib.Path(__file__).parent
        / f"build/mjpc/tasks/panda/{task}/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mj_viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen', width=640, height=480)

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
        def plan(step_limit=1000, cost_limit=None, cost_name=None, viewer=None):
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
                img = mj_viewer.read_pixels(camid=0)
                images.append(img)
                    # total_cost += agent.get_total_cost()
                if cost_name is None:
                    cost = agent.get_total_cost()
                else:
                    cost = agent.get_cost_term_values()[cost_name]
                if i % 20 == 0:
                    # print(i, cost)
                    print(i, f"{agent.get_total_cost():.2f} {cost:.2f}", agent.get_cost_term_values())
                # agent.planner_step()
                if cost_limit is not None and cost <= cost_limit:
                    return True
                # observations.append(environment_step(model, data, actions[-1]))
                # viewer.render()
            return False
        
        def run_once(task_parameters, cost_weights, cost_limit, cost_name, viewer=None):
            if task_parameters is not None:
                agent.set_task_parameters(task_parameters)
            cost_names = agent.get_cost_term_values().keys()
            zeroed_cost_weights = {
                key: cost_weights.get(key, 0.0) for key in cost_names
            }
            agent.set_cost_weights(zeroed_cost_weights)
            return plan(cost_limit=cost_limit, cost_name=cost_name, viewer=viewer)
        
        def run_reset(viewer=None):
            run_once(task_parameters=None, cost_weights={
                "Default Pose": 1
            }, cost_limit=0.02, cost_name="Default Pose", viewer=viewer)

        def run_with_retries(task_name, task_parameters, cost_weights, cost_limit, cost_name, num_retries=3, viewer=None):
            for i in range(num_retries):
                print(f"Task [{task_name}] retry #{i} ...")
                run_reset(viewer=viewer)
                succ = run_once(task_parameters, cost_weights, cost_limit, cost_name, viewer=viewer)
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

        def run_kitchen_cabinet(viewer=None):
            succ = run_with_retries("open microwave", task_parameters={
                "ReachObjectA": "hand",
                "ReachObjectB": "cabinet_doorhandle_l",
                "JointTarget": "leftdoorhinge",
                "JointTargetAngle": 1.0
            }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
            cost_limit=0.02, cost_name="Joint Target", num_retries=3, viewer=viewer)

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
            
            return succ
        
        # def run(viewer=None):
        #     # print(agent.get_parameters())

        #     # for cabinet
        #     # agent.set_task_parameters({"ReachObjectA": "hand"})
        #     # agent.set_task_parameters({"ReachObjectB": "box_right"})
        #     # agent.set_task_parameters({"MoveAwayObjectA": "box_right"})
        #     # agent.set_task_parameters({"MoveAwayObjectB": "doorhandle"})
        #     # agent.set_task_parameters({"MoveAwayDistance": 0.4})
        #     # agent.set_cost_weights(
        #     #     {"Reach": 1, "Joint Target": 0, "Move Away": 1.0}
        #     # )

        #     # for kitchen move kettle
        #     agent.set_task_parameters({"ReachObjectA": "hand"})
        #     agent.set_task_parameters({"ReachObjectB": "kettle_handle"})
        #     agent.set_task_parameters({"FingerTouchObject": "kettle_handle"})
        #     agent.set_task_parameters({"MoveAwayObjectA": "kettle_center"})
        #     agent.set_task_parameters({"MoveAwayObjectB": "microwave_handle"})
        #     agent.set_task_parameters({"MoveAwayDistance": 0.7})
        #     agent.set_cost_weights(
        #         {"Pinch": 0.0, "Finger Touch": 0, "Reach": 0.2, "Reach2": 0.0, "Joint Target": 0, "Move Away": 1, "Default Pose": 0}
        #     )

        #     agent.reset()
        #     environment_reset(model, data)

        #     print(agent.get_task_parameters())
        #     print(agent.get_cost_term_values())
        #     print(agent.get_cost_weights())

        #     # print(data.qpos)
        #     succ = plan(cost_limit=0.01, viewer=viewer, cost_name="Move Away")
        #     # return succ
        #     if not succ:
        #         return False

        #     # default pose
        #     agent.set_cost_weights(
        #         {"Pinch": 0.0, "Finger Touch": 0, "Reach": 0, "Reach2": 0.0, "Joint Target": 0, "Move Away": 0, "Default Pose": 1}
        #     )

        #     # agent.reset()
        #     # environment_reset(model, data)

        #     print(agent.get_task_parameters())
        #     print(agent.get_cost_term_values())
        #     print(agent.get_cost_weights())

        #     # print(data.qpos)
        #     succ = plan(cost_limit=0.01, viewer=viewer, cost_name="Default Pose")
        #     # return succ
        #     if not succ:
        #         return False

        #     # # open cabinet
        #     # agent.set_task_parameters({"ReachObjectA": "hand"})
        #     # agent.set_task_parameters({"ReachObjectB": "doorhandle"})
        #     # agent.set_task_parameters({"JointTarget": "rightdoorhinge"})
        #     # agent.set_task_parameters({"JointTargetAngle": 1.0})
        #     # agent.set_cost_weights(
        #     #     {"Reach": 1, "Joint Target": 0.5, "Move Away": 0.0}
        #     # )

        #     # open microwave
        #     # agent.set_task_parameters({"PinchForce": 50.0})
        #     agent.set_task_parameters({"FingerTouchObject": "microwave_handle"})
        #     agent.set_task_parameters({"ReachObjectA": "hand"})
        #     agent.set_task_parameters({"ReachObjectB": "microwave_handle"})
        #     agent.set_task_parameters({"JointTarget": "micro0joint"})
        #     # agent.set_task_parameters({"ReachObjectB": "cabinet_doorhandle_r"})
        #     # agent.set_task_parameters({"JointTarget": "rightdoorhinge"})
        #     agent.set_task_parameters({"JointTargetAngle": 1.2})
        #     agent.set_cost_weights(
        #         {"Pinch": 0.0, "Finger Touch": 0, "Reach": 1, "Reach2": 0.0, "Joint Target": 1, "Move Away": 0, "Default Pose": 0}
        #     )

        #     print(agent.get_task_parameters())
        #     print(agent.get_total_cost())
        #     print(agent.get_cost_weights())

        #     # print(data.qpos)

        #     succ = plan(cost_limit=0.1, viewer=viewer, cost_name="Joint Target")
        #     if not succ:
        #         return False

        #     # default pose
        #     agent.set_cost_weights(
        #         {"Pinch": 0.0, "Finger Touch": 0, "Reach": 0, "Reach2": 0.0, "Joint Target": 0, "Move Away": 0, "Default Pose": 1}
        #     )

        #     # agent.reset()
        #     # environment_reset(model, data)

        #     print(agent.get_task_parameters())
        #     print(agent.get_cost_term_values())
        #     print(agent.get_cost_weights())

        #     # print(data.qpos)
        #     succ = plan(cost_limit=0.01, viewer=viewer, cost_name="Default Pose")
        #     # return succ
        #     if not succ:
        #         return False
            
        #     # open microwave
        #     # agent.set_task_parameters({"PinchForce": 50.0})
        #     agent.set_task_parameters({"FingerTouchObject": "box"})
        #     agent.set_task_parameters({"ReachObjectA": "hand"})
        #     agent.set_task_parameters({"ReachObjectB": "box"})
        #     agent.set_task_parameters({"Reach2ObjectA": "box"})
        #     agent.set_task_parameters({"Reach2ObjectB": "target_position"})
        #     agent.set_task_parameters({"MoveAwayObjectA": "box"})
        #     agent.set_task_parameters({"MoveAwayObjectB": "microwave_center"})
        #     agent.set_task_parameters({"MoveAwayDistance": 0.7})
        #     agent.set_cost_weights(
        #         {"Pinch": 0.0, "Finger Touch": 0, "Reach": 1, "Reach2": 1.0, "Joint Target": 0, "Move Away": 0, "Default Pose": 0}
        #     )

        #     print(agent.get_task_parameters())
        #     print(agent.get_total_cost())
        #     print(agent.get_cost_weights())

        #     # print(data.qpos)

        #     succ = plan(cost_limit=0.1, viewer=viewer, cost_name="Move Away")
        #     if not succ:
        #         return False

        #     return succ

        if task == "kitchen":
            run_fn = run_kitchen_cabinet
        elif task == "cabinet":
            run_fn = run_cabinet
        else:
            raise NotImplementedError()

        if use_viewer:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                ret = run_fn(viewer=viewer)
        else:
            ret = run_fn(viewer=None)

        images = np.stack(images, 0)
        vidwrite("output.mp4", images, 240)

        return ret


if __name__ == "__main__":
    T = 1
    n_succ = 0
    for _ in range(T):
        n_succ += int(test("cabinet", use_viewer=True))
    
    print(n_succ)
