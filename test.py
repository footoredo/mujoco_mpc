import os
import grpc
from PIL import Image
import mujoco
import mujoco.viewer
import mujoco_mpc
print(mujoco_mpc.__file__)
from mujoco_mpc import agent as agent_lib
import numpy as np


import pathlib


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


def test(use_viewer=True):
    model_path = (
        pathlib.Path(__file__).parent
        / "build/mjpc/tasks/panda/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
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

    with agent_lib.Agent(task_id="Panda", model=model) as agent:
        def plan(step_limit=5000, cost_limit=None, viewer=None):
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
                    # total_cost += agent.get_total_cost()
                cost = agent.get_total_cost()
                if i % 20 == 0:
                    print(i, cost)
                # agent.planner_step()
                if cost_limit is not None and cost <= cost_limit:
                    return True
                # observations.append(environment_step(model, data, actions[-1]))
                # viewer.render()
            return False
        
        def run(viewer=None):
            # print(agent.get_parameters())
            agent.set_task_parameters({"ReachObjectA": "hand"})
            agent.set_task_parameters({"ReachObjectB": "box_right"})
            agent.set_task_parameters({"MoveAwayObjectA": "box_right"})
            agent.set_task_parameters({"MoveAwayObjectB": "doorhandle"})
            agent.set_task_parameters({"MoveAwayDistance": 0.4})
            agent.set_cost_weights(
                {"Reach": 1, "Joint Target": 0, "Move Away": 1.0}
            )

            agent.reset()
            environment_reset(model, data)

            print(agent.get_task_parameters())
            print(agent.get_total_cost())
            print(agent.get_cost_weights())

            # print(data.qpos)
            succ = plan(cost_limit=0.01, viewer=viewer)
            if not succ:
                return False

            agent.set_task_parameters({"ReachObjectA": "hand"})
            agent.set_task_parameters({"ReachObjectB": "doorhandle"})
            agent.set_task_parameters({"JointTarget": "rightdoorhinge"})
            agent.set_task_parameters({"JointTargetAngle": 1.0})
            agent.set_cost_weights(
                {"Reach": 1, "Joint Target": 0.5, "Move Away": 0.0}
            )

            print(agent.get_task_parameters())
            print(agent.get_total_cost())
            print(agent.get_cost_weights())

            # print(data.qpos)

            succ = plan(cost_limit=0.1, viewer=viewer)
            return succ

        if use_viewer:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                return run(viewer=viewer)
        else:
            return run(viewer=None)


if __name__ == "__main__":
    T = 1
    n_succ = 0
    for _ in range(T):
        n_succ += int(test(use_viewer=True))
    
    print(n_succ)
