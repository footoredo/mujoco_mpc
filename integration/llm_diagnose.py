import os
import sys
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


if __name__ == "__main__":
    env = sys.argv[1]
    data_path = sys.argv[2]

    data = joblib.load(data_path)

    model_path = (
        pathlib.Path(__file__).parent.parent
        / f"build/mjpc/tasks/panda/{env}/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    
    while True:
        type, name = input().split()
        try:
            if type == "j":
                print(data.joint(name).qpos[0])
            else:
                print(data.site(name).xpos)
        except:
            pass