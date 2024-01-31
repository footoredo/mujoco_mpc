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
from scipy.spatial.transform import Rotation as R
import requests
import copy


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
#   print(data.joint("rightdoorhandle").qpos)
  if ENV == "cabinet" and OPENED_CABINET:
    data.qpos[15] = 1.57
  if ENV == "locklock" and get_joint_value("red_switch_handle_joint") < 1.5:
    data.joint("rightdoorhinge").qpos[0] = 0.
  if ENV == "cabinet" and NO_LOCK:
    data.qpos[0] = 100
  if ENV == "long":
    # data.joint("leftdoorhinge").qpos[0] = -1.57
    distance = get_object_distance("weight", "weightsensor")
    if distance > 0.05:
        data.joint("slidedoor_joint").qpos[0] = 0.
  return get_observation(model, data)


def environment_reset(model, data):
  mujoco.mj_resetData(model, data)
  return get_observation(model, data)

REAL_ROBOT = False

ENV = "blocks"
REPEATS = 2
RETRIES = 2
# ENV = "cabinet"
# ENV = "kitchen"
SAVE_VIDEO = False
OPENED_CABINET = False
NO_LOCK = False
IS_COP = False

REWARD_CNT = {
    "min_l2": 0,
    "max_l2": 0,
    "joint": 0,
    "pinch": 0,
    "stack": 0,
    "lift": 0
}

TASK_PARAMS = {}
COST_WEIGHTS = {}

BLOCKS_NAME_MAPPING = {
    "palm": "hand",
    "yellow_block": "yellow_block",
    "red_block": "red_block",
    "blue_block": "blue_block",
    "right_cube": "yellow_block",
    "rightside_cube": "yellow_block",
    "left_cube": "red_block",
    "leftside_cube": "red_block",
    "crate": "red_bin",
    "plate": "red_bin",
    "square_plate": "red_bin"
}

BLOCKS_SITE_MAPPING = {
    "hand": "pinch",
    "yellow_block": "yellow_block",
    "red_block": "red_block",
    "blue_block": "blue_block",
    "red_bin": "red_bin"
}

LOCKLOCK_NAME_MAPPING = {
    "palm": "hand",
    "wooden_cabinet": "cabinet",
    "wooden_cabinet_door_handle": "rightdoorhandle",
    "wooden_cabinet_door": "rightdoorhinge",
    "wooden_cabinet_door_hinge": "rightdoorhinge",
    "wooden_cabinet_inside": "right_target_position",
    "blue_cube": "blue_block",
    "red_lever_handle": "red_switch_handle",
    "red_lever": "red_switch_handle_joint",
    "red_lever_joint": "red_switch_handle_joint"
}

LOCKLOCK_SITE_MAPPING = {
    "hand": "eeff",
    "rightdoorhinge": "rightdoorhinge",
    "leftdoorhinge": "leftdoorhinge",
    "rightdoorhandle": "rightdoor_site",
    "leftdoorhandle": "leftdoor_site"
}

LONG_NAME_MAPPING = {
    "palm": "hand",
    "weight_sensor_lock": "weightsensor",
    "weight": "weight",
    "green_weight_sensor": "weightsensor",
    "green_weight_sensor_lock": "weightsensor",
    "green_weight": "weight",
    "red_weight": "weight",
    "red_cube": "weight",
    "hinge_cabinet_door_handle": "leftdoorhandle",
    "hinge_cabinet": "leftdoorhinge",
    "top_cabinet_door_handle": "leftdoorhandle",
    "top_cabinet": "leftdoorhinge",
    "wooden_cabinet_door_handle": "leftdoorhandle",
    "wooden_cabinet": "leftdoorhinge",
    "red_block_right_side": "barright",
    "blue_block_right_side": "barright",
    # "hinge_cabinet_door_handle": "rightdoorhandle",
    # "hinge_cabinet": "rightdoorhinge",
    "slide_cabinet_door_handle": "slidedoorhandle",
    "slide_cabinet": "slidedoorjoint",
    "bottom_cabinet_door_handle": "slidedoorhandle",
    "bottom_cabinet": "slidedoorjoint",
    "stone_cabinet_door_handle": "slidedoorhandle",
    "stone_cabinet": "slidedoorjoint",
    "hinge_cabinet_inside": "leftcabinetinside",
    "microwave": "micro0joint",
}

LONG_SITE_MAPPING = {
    "hand": "eeff",
    "rightdoorhandle": "rightdoor_site",
    "leftdoorhandle": "leftdoor_site",
    "slidedoorhandle": "slide_site",
    "slidedoorjoint": "slidedoor_joint",
    "microwave_handle": "microhandle_site",
}

CABINET_NAME_MAPPING = {
    "palm": "hand",
    "red_block": "box",
    "red_block_left_side": "box_left",
    "red_block_right_side": "box_right",
    "red_bar": "box",
    "red_bar_left_side": "box_left",
    "red_bar_right_side": "box_right",
    "yellow_cube": "target",
    # "left_wooden_cabinet_handle": "leftdoorhandle",
    # "right_wooden_cabinet_handle": "rightdoorhandle",
    # "left_wooden_cabinet": "leftdoorhinge",
    # "right_wooden_cabinet": "rightdoorhinge",
    # "right_wooden_cabinet_inside": "right_target_position",
    # "left_wooden_cabinet_inside": "left_target_position"
    "wooden_cabinet_handle": "rightdoorhandle",
    "wooden_cabinet": "rightdoorhinge",
    "wooden_cabinet_inside": "right_target_position",
    "target_position_in_wooden_cabinet": "right_target_position",
    "target_position": "right_target_position"
}

CABINET_SITE_MAPPING = {
    "hand": "eeff",
    "rightdoorhandle": "rightdoor_site",
    "leftdoorhandle": "leftdoor_site"
}

KITCHEN_NAME_MAPPING = {
    "palm": "hand",
    "cabinet_handle": "cabinet_doorhandle_l",
    "right_cabinet_handle": "cabinet_doorhandle_r",
    "cabinet": "leftdoorhinge",
    "right_cabinet": "rightdoorhinge",
    "microwave_handle": "microwave_handle",
    "microwave": "micro0joint",
    "blue_kettle_handle": "kettle_handle",
    "green_apple": "box",
    "green_cube": "box",
    "target_position": "target_position"
}

KITCHEN_SITE_MAPPING = {
    "cabinet_doorhandle_r": "rightdoor_site",
    "cabinet_doorhandle_l": "leftdoor_site",
    "kettle_handle": "kettle_site0",
    "microwave_handle": "microhandle_site",
    "hand": "eeff"
}

NAME_MAPPING = {
    "locklock": LOCKLOCK_NAME_MAPPING,
    "blocks": BLOCKS_NAME_MAPPING,
    "cabinet": CABINET_NAME_MAPPING,
    "kitchen": KITCHEN_NAME_MAPPING,
    "long": LONG_NAME_MAPPING,
}

SITE_MAPPING = {
    "locklock": LOCKLOCK_SITE_MAPPING,
    "cabinet": CABINET_SITE_MAPPING,
    "kitchen": KITCHEN_SITE_MAPPING,
    "blocks": BLOCKS_SITE_MAPPING,
    "long": LONG_SITE_MAPPING
}

PRIMARY_REWARD = None

COST_NAMES_REQUIRED = []


def set_env(env):
    global ENV
    ENV = env
    
    
def set_repeats(repeats):
    global REPEATS
    REPEATS = repeats
    
    
def set_retries(retries):
    global RETRIES
    RETRIES = retries


def set_runner(runner):
    global RUNNER
    RUNNER = runner


def reset_reward():
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    for key in REWARD_CNT.keys():
        REWARD_CNT[key] = 0
    TASK_PARAMS = {}
    COST_WEIGHTS = {}
    PRIMARY_REWARD = None
    COST_NAMES_REQUIRED = []
    print(0)
    # COST_WEIGHTS["Safety"] = 0.1
    # COST_NAMES_REQUIRED.append("Safety")
    # REWARD_CNT["Safety"] = 1
    # COST_WEIGHTS["HitGround"] = 0
    COST_WEIGHTS["LockBin"] = 1
    COST_WEIGHTS["BlockOrient"] = 1
    COST_WEIGHTS["OpenGripper"] = 1



def map_name(name):
    return NAME_MAPPING[ENV].get(name, name)


def map_site(name):
    return SITE_MAPPING[ENV].get(name, name)


def get_object_position(obj_name):
    mapped_name = map_name(obj_name)
    site_name = map_site(mapped_name)

    try:
        return RUNNER.data.site(site_name).xpos.copy()
    except:
        return None


def get_object_distance(obj1, obj2):
    pos1 = get_object_position(obj1)
    pos2 = get_object_position(obj2)
    
    if pos1 is None or pos2 is None:
        return None
    
    return np.linalg.norm(pos1 - pos2)


def get_joint_value(obj):
    mapped_name = map_name(obj)
    site_name = map_site(mapped_name)
    
    return RUNNER.data.joint(site_name).qpos[0]


def is_joint(obj):
    name = map_name(obj)
    return name.endswith("joint") or name.endswith("hinge")


def lift(obj, height=1.0, primary_reward=False):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    if REWARD_CNT["lift"] >= 1:
        return
    if is_joint(obj):
        return
    REWARD_CNT["lift"] += 1
    cnt = REWARD_CNT["lift"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"Lift{cnt}Object"] = map_name(obj)
    TASK_PARAMS[f"Lift{cnt}Height"] = height
    COST_WEIGHTS[f"Lift{cnt}"] = 1.0
    # print(COST_WEIGHTS)

    if primary_reward:
        PRIMARY_REWARD = f"Lift{cnt}"

    COST_NAMES_REQUIRED.append(f"Lift{cnt}")


def pinch_finger(obj, primary_reward=False):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    if REWARD_CNT["pinch"] >= 1:
        return
    if is_joint(obj):
        return
    REWARD_CNT["pinch"] += 1
    cnt = REWARD_CNT["pinch"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"FingerTouch{cnt}Object"] = map_name(obj)
    COST_WEIGHTS[f"Pinch{cnt}"] = 1.0
    # print(COST_WEIGHTS)

    if primary_reward:
        PRIMARY_REWARD = f"Pinch{cnt}"

    COST_NAMES_REQUIRED.append(f"Pinch{cnt}")


def stack_reward(obj1, obj2, primary_reward=False):  # stack obj1 on top of obj2
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    if REWARD_CNT["stack"] >= 1:
        return
    if is_joint(obj1) or is_joint(obj2):
        return
    REWARD_CNT["stack"] += 1
    cnt = REWARD_CNT["stack"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"Stack{cnt}ObjectA"] = map_name(obj1)
    TASK_PARAMS[f"Stack{cnt}ObjectB"] = map_name(obj2)
    COST_WEIGHTS[f"Stack{cnt}"] = 1.0

    if primary_reward:
        PRIMARY_REWARD = f"Stack{cnt}"

    COST_NAMES_REQUIRED.append(f"Stack{cnt}")


def minimize_l2_distance_reward(obj1, obj2, primary_reward=False):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    if REWARD_CNT["min_l2"] >= 3:
        return
    if is_joint(obj1) or is_joint(obj2):
        return
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
    if REWARD_CNT["max_l2"] >= 1:
        return
    if is_joint(obj1) or is_joint(obj2):
        return
    original_distance = get_object_distance(obj1, obj2)
    if original_distance is None:
        return
    print("original_distance:", original_distance)
    REWARD_CNT["max_l2"] += 1
    cnt = REWARD_CNT["max_l2"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"MoveAway{cnt}ObjectA"] = map_name(obj1)
    TASK_PARAMS[f"MoveAway{cnt}ObjectB"] = map_name(obj2)
    # TASK_PARAMS[f"MoveAwayDistance"] = distance * 1.5 if ENV == "kitchen" else distance * 0.8
    move_away_distance = original_distance + 0.6 if ENV == "kitchen" else original_distance + 0.3
    if ENV == "kitchen":
        max_distance = 2.0
    elif ENV == "cabinet":
        max_distance = 0.8
    else:
        max_distance = 1.0
    print(move_away_distance, max_distance)
    move_away_distance = min(move_away_distance, max_distance)
    TASK_PARAMS[f"MoveAwayDistance"] = move_away_distance
    COST_WEIGHTS[f"Move Away{cnt}"] = 1.0

    if primary_reward:
        PRIMARY_REWARD = f"Move Away{cnt}"

    COST_NAMES_REQUIRED.append(f"Move Away{cnt}")


def set_joint_fraction_reward(obj, fraction, primary_reward=False):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED
    if ENV == "blocks":  # no joint for blocks
        return
    if REWARD_CNT["joint"] >= 1:
        return
    if not is_joint(obj):
        return
    REWARD_CNT["joint"] += 1
    cnt = REWARD_CNT["joint"]
    if cnt == 1:
        cnt = ""
    TASK_PARAMS[f"JointTarget{cnt}"] = map_name(obj)
    if map_name(obj) == "slidedoorjoint":
        fraction = fraction * 0.4
    else:
        fraction = fraction * 1.5
    TASK_PARAMS[f"JointTarget{cnt}Angle"] = fraction
    COST_WEIGHTS[f"Joint Target{cnt}"] = 1.0

    if primary_reward:
        PRIMARY_REWARD = f"Joint Target{cnt}"

    COST_NAMES_REQUIRED.append(f"Joint Target{cnt}")


def set_primary_reward(reward_index):
    global PRIMARY_REWARD
    PRIMARY_REWARD = COST_NAMES_REQUIRED[reward_index]


last_primary = None


class Runner:
    def __init__(self, task="kitchen", use_viewer=True, init_data=None, save_video=True, save_last_img=False, verbose=False):
        self.task = task
        self.use_viewer = use_viewer
        self.init_data = init_data
        self.save_video = save_video
        self.save_last_img = save_last_img
        self.verbose = verbose

        model_path = (
            pathlib.Path(__file__).parent.parent
            / f"build/mjpc/tasks/panda/{task}/task.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if init_data is None:
            self.data = mujoco.MjData(self.model)
        else:
            self.data = init_data
            
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # print(self.data.site("eeff").xpos)

        # self.mj_viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen', width=1280, height=960)
        self.mj_viewer = None

        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.repeats = REPEATS

        self.actions = []
        self.observations = []

        self.images = []

        self.agent = agent_lib.Agent(task_id=f"Panda {self.task.capitalize()}", model=self.model)

    # def get_joints(self):
    #     joints = []
    #     for i in range(8):
    #         joints.append(self.data.joint(f"joint{i}"))

    def plan(self, step_limit=500, cost_limit=None, cost_name=None):
        global last_primary
        if cost_name is not None and cost_name.startswith("Default Pose"):
            last_primary = None
        else:
            last_primary = cost_name
        num_steps = 0
        best_cost = 1e9
        last_updated = 0
        satisfied = 0
        waypoints = []
        
        franka_ee_pos = self.data.site("franka_ee").xpos.copy()
        pinch_pos = self.data.site("pinch").xpos.copy()
        
        print(pinch_pos - franka_ee_pos)
        franka_diff = pinch_pos - franka_ee_pos
        
        while True:
            reach_cost = self.agent.get_cost_term_values()["Reach"]
            pinch_position = self.data.site('pinch').xpos.copy()
            franka_ee_position = self.data.site('franka_ee').xpos.copy()
            # print(franka_ee_position)
            # if reach_cost > 0.2:
            #     step_size = 50
            # elif reach_cost > 0.03:
            #     step_size = 20
            # else:
            #     step_size = 10
            update_step_size = 50
            # waypoint_step_size = 10
            # step_size = 50
            waypoint_step_size = 10 if reach_cost < 0.2 else 50
            if num_steps % waypoint_step_size == 0:
                # print(site_translation)
                # site_translation = site_translation - np.array([0.1, 0.0, 0.0])  # from pinch to franka ee pos
                site_translation = franka_ee_position
                # site_translation = pinch_position - franka_diff
                # print(site_translation - franka_ee_position)
                # site_rotation = R.from_matrix(self.data.site('pinch').xmat.copy().reshape(3, 3))
                site_rotation = R.from_matrix(self.data.site('franka_ee').xmat.copy().reshape(3, 3))
                # site_rotation = site_rotation.as_rotvec() - np.array([0.0, 1.5708, 0.0])
                rotation_offset = R.from_quat([1, 0, 0, 0])
                site_rotation = rotation_offset.inv() * site_rotation 
                print("rotation", site_rotation.as_rotvec(), site_rotation.as_quat())
                print(site_rotation.as_matrix())
                site_rotation = site_rotation.as_matrix()
                # site_rotation = np.array([0.0, 0.0, 0.0])  # ignore rotation for now
                gripper_pose = self.data.joint("right_driver_joint").qpos  # 0.0 is fully open, 0.8 is fully closed
                gripper_pose = max(min(int(gripper_pose / 0.8 * 255), 255), 0)  # to robotiq pose

                if num_steps > 0:
                    data_to_send = {
                        "translation": site_translation.tolist(),
                        "rotation": site_rotation.flatten().tolist(),  # [3x3] -> [9]
                        "gripper_pose": gripper_pose,
                        "qpos": self.data.qpos.tolist(),
                        "type": "update"
                    }
                    waypoints.append(copy.deepcopy(data_to_send))
                    
                else:
                    data_to_send = {
                        "type": "init"
                    }
            
            if num_steps % update_step_size == 0:
                    
                print("Waiting for obs...")
                
                data_to_send = waypoints

                # Make the POST request
                
                if REAL_ROBOT:
                    response = requests.post("http://localhost:5000/act_ret_obs", json=data_to_send)
                    if response.status_code == 200:
                        received_data = response.json()
                        # Process the received data if necessary
                        print("Received data:", received_data)
                    else:
                        print("Error in POST request:", response.status_code)
        
                    # joint_pos = received_data["joints"]
                    # ee_pos = received_data["ee"]
                    # print(ee_pos - franka_ee_position)
                    joint_pos = self.data.qpos[-15:-8]
                else:
                    print("Not receiving data!")
                    received_data = {}
                    joint_pos = self.data.qpos[-15:-8]
                obj_pos = self.data.qpos[:-15]
                gripper_pos = self.data.qpos[-8:]
                
                if "objects" in received_data and received_data["objects"] is not None:
                    objects_data = received_data["objects"]
                    objects_poses = {}
                    for _, obj in objects_data["objects"].items():
                        obj_name = obj["object_type"]
                        obj_pose_hmat = np.array(obj["pose"])
                        obj_translation = obj_pose_hmat[:3, 3].copy()
                        obj_orientation = obj_pose_hmat[:3, :3].copy()
                        objects_poses[obj_name] = {
                            "translation": obj_translation,
                            "orientation": obj_orientation
                        }
                    print(objects_data)
                    obj_pos[:3] = objects_poses["lemon"]["translation"]
                    obj_pos[3:7] = R.from_matrix(objects_poses["lemon"]["orientation"]).as_quat()
                    obj_pos[7:10] = objects_poses["apple"]["translation"]
                    # obj_pos[10:14] = R.from_matrix(objects_poses["apple"]["orientation"]).as_quat()
                    obj_pos[21:24] = objects_poses["bowl"]["translation"]
                    # obj_pos[24:28] = R.from_matrix(objects_poses["bowl"]["orientation"]).as_quat()
                # gripper_pos = received_data["gripper"]
                # gripper_pos = gripper_pos / 255 * 0.8  # robotiq to mujoco
                # gripper_pos = 0.0
                
                qpos = np.concatenate((obj_pos, joint_pos, gripper_pos))

                self.data.qpos[:] = qpos
                self.data.qvel[:] = 0.
                
                # print(self.data.qpos)
                
                if self.viewer is not None:
                    self.viewer.sync()
                
                if False:
                    input("Continue? ")
                    
                waypoints = []
                
                # self.actions.append(site_action)
            # else:
            self.agent.set_state(
                time=self.data.time,
                qpos=self.data.qpos,
                qvel=self.data.qvel,
                act=self.data.act,
                mocap_pos=self.data.mocap_pos,
                mocap_quat=self.data.mocap_quat,
                userdata=self.data.userdata,
            )
            # print(self.data.qpos)
            reach2_cost = self.agent.get_cost_term_values()["Reach2"]
            reach_cost = self.agent.get_cost_term_values()["Reach"]
            # if reach1_cost > 0.02:
            #     self.actions[-1][-1] = 0.
            lift_cost = self.agent.get_cost_term_values()["Lift"]
            # self.agent.set_cost_weights({
            #     "Reach2": lift_cost < 0.05 or reach2_cost <= 0.1,
            #     "Lift": reach2_cost > 0.1,
            #     # "BlockOrient": reach2_cost > 0.09
            #     "BlockOrient": lift_cost > 0.08 and reach2_cost > 0.1,
            #     "Reach": reach2_cost > 0.03,
            #     # "OpenGripper": reach_cost > 0.05
            # })
            self.agent.planner_step()
            # print(i)
            # actions.append(agent.get_action(averaging_duration=control_timestep))
            self.actions.append(self.agent.get_action())
            # print(i)
            # satisfied = False
            for _ in range(self.repeats):
                
                self.observations.append(environment_step(self.model, self.data, self.actions[-1]))
                # print("hand xpos:", data.site("eeff").xpos)
                if self.viewer is not None:
                    self.viewer.sync()

                if cost_name is None:
                    cost = self.agent.get_total_cost()
                else:
                    cost = self.agent.get_cost_term_values()[cost_name]
                if cost < best_cost - 1e-2:
                    best_cost = cost
                    last_updated = 0
                # agent.planner_step()
                if num_steps > 20:
                    if cost_limit is not None and cost <= cost_limit:
                        satisfied += 1

            print("Cost:", self.agent.get_cost_term_values())
            print("Cost weights:", self.agent.get_cost_weights())

            if self.save_video:
                img = self.mj_viewer.read_pixels(camid=0)
                self.images.append(img)
                # total_cost += agent.get_total_cost()
            
            if num_steps % 20 == 0 and self.verbose:
                # print(i, cost)
                print(num_steps, f"{self.agent.get_total_cost():.2f} {cost:.2f}", self.agent.get_cost_term_values())

            if satisfied:
                return True

            # observations.append(environment_step(model, data, actions[-1]))
            # viewer.render()
            num_steps += 1
            last_updated += 1
            # print(last_updated, best_cost)
            if last_updated > 200 and num_steps > step_limit:
                break
        return False
        
    def run_once(self, task_parameters, cost_weights, cost_limit, cost_name=None, step_limit=1000):
        if task_parameters is not None:
            self.agent.set_task_parameters(task_parameters)
        cost_names = self.agent.get_cost_term_values().keys()
        zeroed_cost_weights = {
            key: cost_weights.get(key, 0.0) for key in cost_names
        }
        self.agent.set_cost_weights(zeroed_cost_weights)
        return self.plan(cost_limit=cost_limit, cost_name=cost_name, step_limit=step_limit)
        
    def run_reset_cartisian(self):
        for _ in range(500):
            environment_step(self.model, self.data, np.zeros(self.model.nu))
            if self.viewer is not None:
                self.viewer.sync()
        
    def run_reset(self):
        # print(last_primary)
        succ = self.run_once(task_parameters=None, cost_weights={
            "Default Pose": 1, last_primary: 1
        }, cost_limit=0.02, cost_name="Default Pose", step_limit=200)
        return succ

    def run_reset_no_obstruction(self):
        succ = self.run_once(task_parameters=None, cost_weights={
            "Default Pose No-Obstruction": 1, last_primary: 1
        }, cost_limit=0.02, cost_name="Default Pose No-Obstruction", step_limit=200)
        return succ

    def run_with_retries(self, task_name, task_parameters, cost_weights, cost_limit, cost_name=None, num_retries=5, step_limit=1000):
        retries = 0
        best_primary_cost = 1e9
        
        # if len(cost_weights) == 0:
        #     return True
        # elif len(cost_weights) == 1:
        #     if "Reach" in cost_weights:
        #         if task_parameters["ReachObjectA"] == "hand" or task_parameters["ReachObjectB"] == "hand":
        #             return True
        #         else:
        #             return False
        #     else:
        #         return False
        
        while True:
            print(f"Task [{task_name}] retry #{retries} ...", flush=True)
            # self.run_reset()
            self.run_reset_cartisian()
            costs = self.agent.get_cost_term_values()
            # if PRIMARY_REWARD is not None:
            #     primary_cost_before = costs[PRIMARY_REWARD]
            improved = False
            succ = self.run_once(task_parameters, cost_weights, cost_limit, cost_name=cost_name, step_limit=step_limit)
            costs = self.agent.get_cost_term_values()
            for name in sorted(cost_weights.keys()):
                print(name, costs[name], flush=True)
            print(flush=True)
            if PRIMARY_REWARD is not None:
                primary_cost_after = costs[PRIMARY_REWARD]
                if best_primary_cost - primary_cost_after > 0.05:
                    improved = True
                    best_primary_cost = primary_cost_after
            if succ:
                return True
            retries += 1
            if retries >= num_retries and not improved:
                break
        return False
        
    def run_kitchen(self):
        succ = self.run_with_retries("move kettle", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "kettle_handle",
            "MoveAwayObjectA": "kettle_center",
            "MoveAwayObjectB": "microwave_handle",
            "MoveAwayDistance": 0.7
        }, cost_weights={"Reach": 1.0, "Move Away": 1.0},
        cost_limit=0.02, cost_name="Move Away", num_retries=3)

        if not succ:
            return False

        succ = self.run_with_retries("open microwave", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "microwave_handle",
            "JointTarget": "micro0joint",
            "JointTargetAngle": 1.2
        }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
        cost_limit=0.02, cost_name="Joint Target", num_retries=3)

        if not succ:
            return False

        succ = self.run_with_retries("move cube", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "box",
            "Reach2ObjectA": "box",
            "Reach2ObjectB": "target_position"
        }, cost_weights={"Reach": 1.0, "Reach2": 1.0},
        cost_limit=0.02, cost_name="Reach2", num_retries=3)

        return succ
        
    def run_kitchen_cabinet_l(self):
        succ = self.run_with_retries("open cabinet_l", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "cabinet_doorhandle_l",
            "JointTarget": "leftdoorhinge",
            "JointTargetAngle": 1.5
        }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
        cost_limit=0.02, cost_name="Joint Target", num_retries=3)

        return succ

    def run_kitchen_cabinet_r(self):
        succ = self.run_with_retries("open cabinet_r", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "cabinet_doorhandle_r",
            "JointTarget": "rightdoorhinge",
            "JointTargetAngle": 1.5
        }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
        cost_limit=0.02, cost_name="Joint Target", num_retries=3)

        return succ
        
    def run_kitchen_cabinet_both(self):
        succ = self.run_kitchen_cabinet_r()
        if not succ:
            return False
        succ = self.run_kitchen_cabinet_l()
        # self.run_reset_no_obstruction()
        return succ

    def run_kitchen_dummy(self):
        succ = self.run_reset()
        return succ

    def run_cabinet(self):
        succ = self.run_with_retries("remove stick", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "box_right",
            "MoveAwayObjectA": "box_right",
            "MoveAwayObjectB": "rightdoorhandle",
            "MoveAwayDistance": 0.6
        }, cost_weights={"Reach": 1.0, "Move Away": 1.0},
        cost_limit=0.02, cost_name="Move Away", num_retries=3)

        if not succ:
            return False
        
        succ = self.run_with_retries("open door", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "rightdoorhandle",
            "JointTarget": "rightdoorhinge",
            "JointTargetAngle": 1.5
        }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
        cost_limit=0.02, cost_name="Joint Target", num_retries=3)

        if not succ:
            return False
        
        succ = self.run_with_retries("move cube", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "target",
            "Reach2ObjectA": "target",
            "Reach2ObjectB": "target_position"
        }, cost_weights={"Reach": 1.0, "Reach2": 1.0},
        cost_limit=0.02, cost_name="Reach2", num_retries=3)

        return succ
        
    def run_cabinet_test1(self):
        succ = self.run_with_retries("remove stick", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "box_right",
            "MoveAwayObjectA": "box_right",
            "MoveAwayObjectB": "rightdoorhandle",
            "MoveAwayDistance": 0.6
        }, cost_weights={"Reach": 1.0, "Move Away": 1.0},
        cost_limit=0.02, cost_name="Move Away", num_retries=3)

        return succ
        
    def run_cabinet_test2(self):
        succ = self.run_with_retries("open door", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "rightdoorhandle",
            "JointTarget": "rightdoorhinge",
            "JointTargetAngle": 1.5
        }, cost_weights={"Reach": 1.0, "Joint Target": 1.0},
        cost_limit=0.02, cost_name="Joint Target", num_retries=3)

        return succ
    
    def run_cabinet_move_cube(self):
        succ = self.run_with_retries("move cube", task_parameters={
            "ReachObjectA": "hand",
            "ReachObjectB": "target",
            "Reach2ObjectA": "target",
            "Reach2ObjectB": "right_target_position"
        }, cost_weights={"Reach": 1.0, "Reach2": 1.0},
        cost_limit=0.02, cost_name="Reach2", num_retries=3)

        return succ

    def run_cabinet_dummy(self):
        succ = self.run_with_retries("remove stick", task_parameters={
            "ReachObjectA": "rightdoorhandle",
            "ReachObjectB": "leftdoorhandle",
        }, cost_weights={"Reach": 1.0},
        cost_limit=0.02, cost_name="Reach", num_retries=1, step_limit=50)
        input()

        return succ

    def run_custom(self):
        if IS_COP:
            cost_limit = 0.002
        else:
            if PRIMARY_REWARD is None:
                cost_limit = 0.01 * sum(list(COST_WEIGHTS.values()))
            else:
                cost_limit = 0.02
        succ = self.run_with_retries("custom", task_parameters=TASK_PARAMS, cost_weights=COST_WEIGHTS,
        cost_limit=cost_limit, cost_name=PRIMARY_REWARD, num_retries=RETRIES, step_limit=500)

        return succ
    
    def check_goal_completed(self):
        if self.task == "kitchen":
            microwave_joint = self.data.joint("micro0joint").qpos[0]
            # print(microwave_joint)
            return microwave_joint < -1.4
        elif self.task == "cabinet":
            distance = get_object_distance("right_target_position", "yellow_cube")
            return distance < 0.05
        elif self.task == "locklock":
            # distance = get_object_distance("wooden_cabinet_inside", "blue_cube")
            door_hinge = self.data.joint("rightdoorhinge").qpos[0]
            return door_hinge > 1.4
        elif self.task == "blocks":
            distance1 = get_object_distance("crate", "red_block")
            distance2 = get_object_distance("crate", "yellow_block")
            if distance2 < 0.05:
                return 1
            elif distance1 < 0.05:
                return -1
            else:
                return 0
        elif self.task == "long":
            slide_joint = self.data.joint("slidedoor_joint").qpos[0]
            return slide_joint > 0.35
        else:
            return 0
    
    def execute(self, custom=False, reset_after_done=True):
        if custom:
            run_fn = self.run_custom
        else:
            if self.task == "kitchen":
                # run_fn = self.run_kitchen_cabinet_both
                # run_fn = self.run_kitchen_dummy
                run_fn = self.run_reset_no_obstruction
            elif self.task == "cabinet":
                run_fn = self.run_reset_no_obstruction
                # run_fn = self.run_cabinet_move_cube
            else:
                run_fn = self.run_reset_no_obstruction
            
        self.outcome = run_fn()
        self.goal_completed = self.check_goal_completed()

        # if not custom:
        #     input()
        
        if reset_after_done:
            self.run_reset_no_obstruction()

        return self.outcome

    def finish(self):
        self.agent.reset()
        if self.viewer is not None:
            self.viewer.close()
        
        if self.save_video:
            images = np.stack(self.images, 0)
            vidwrite("output.mp4", images, 240 * 5)
            
        # print(111)

        if self.save_last_img and self.mj_viewer is not None:
            im = self.mj_viewer.read_pixels(camid=0)
            cv2.imwrite('output.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        joblib.dump(self.data, "data.joblib")

        with open("outcome", "w") as outcome_f:
            outcome_f.write(str(int(self.outcome)))
            
        with open("goal_completed", "w") as gc_f:
            gc_f.write(str(int(self.goal_completed)))


RUNNER = None


def runner_init(load_init=True):
    if load_init:
        try:
            init_data = joblib.load("data.joblib")
        except FileNotFoundError:
            init_data = None
    else:
        init_data = None

    global RUNNER
    RUNNER = Runner(ENV, use_viewer=True, save_video=SAVE_VIDEO, save_last_img=True, init_data=init_data)


def cop_runner_init():
    global IS_COP
    IS_COP = True
    runner_init(load_init=False)


def execute_plan(duration=None, finish=True, reset_after_done=True):
    print(TASK_PARAMS)
    print(COST_WEIGHTS)
    print(PRIMARY_REWARD)
    
    print(int(RUNNER.execute(custom=True, reset_after_done=reset_after_done)))
    if finish:
        RUNNER.finish()
        
    print("Done")


def end_effector_to(position):
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED

    reset_reward()
    TASK_PARAMS["EndEffectorTo1"] = position[0]
    TASK_PARAMS["EndEffectorTo2"] = position[1]
    TASK_PARAMS["EndEffectorTo3"] = position[2]
    COST_WEIGHTS["End-Effector To"] = 1.
    PRIMARY_REWARD = "End-Effector To"

    COST_WEIGHTS["Control"] = 3.

    execute_plan(finish=False, reset_after_done=False)


def end_effector_open():
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED

    reset_reward()
    COST_WEIGHTS["End-Effector Open"] = 1.
    PRIMARY_REWARD = "End-Effector Open"

    execute_plan(finish=False, reset_after_done=False)


def end_effector_close():
    global REWARD_CNT, TASK_PARAMS, COST_WEIGHTS, PRIMARY_REWARD, COST_NAMES_REQUIRED

    reset_reward()
    COST_WEIGHTS["End-Effector Close"] = 1.
    PRIMARY_REWARD = "End-Effector Close"

    execute_plan(finish=False, reset_after_done=False)


def finish():
    RUNNER.finish()


def run_dummy():
    RUNNER.execute(custom=False, reset_after_done=False)


if __name__ == "__main__":

    init_data = None

    # try:
    #     init_data = joblib.load("data.joblib")
    # except FileNotFoundError:
    #     init_data = None

    RUNNER = Runner("kitchen", use_viewer=True, save_video=False, save_last_img=True, init_data=init_data)

    T = 1
    n_succ = 0
    for _ in range(T):
        n_succ += int(RUNNER.execute(custom=False, reset_after_done=False))
    
    print(n_succ)
