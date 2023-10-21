// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/panda/kitchen/kitchen.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace panda {
std::string Kitchen::XmlPath() const {
  return GetModelPath("panda/kitchen/task.xml");
}
std::string Kitchen::Name() const { return "Panda Kitchen"; }
const std::array<std::string, 14> object_names = {
	"hand", "box", "target_position", "cabinet_doorhandle_r", 
  "cabinet_doorhandle_l", "kettle_handle", "kettle_center", 
  "microwave_handle", "microwave_center", "slide_handle", 
  "knob1", "knob2", "knob3", "knob4"
};

const std::array<std::string, 8> joint_names = {
	"leftdoorhinge", "rightdoorhinge", "knob1_joint", "knob2_joint", "knob3_joint", "knob4_joint", "lightswitch_joint", "micro0joint"
};


// ---------- Residuals for in-panda manipulation task ---------
//   Number of residuals: 5
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): cube angular velocity
//     Residual (4): control
// ------------------------------------------------------------
void Kitchen::ResidualFn::Residual(const mjModel* model, const mjData* data,
                     double* residual) const {

  int counter = 0;
  int param_counter = 0;

  // double* left_finger_force = SensorByName(model, data, "leftfingerforce");
  // double* right_finger_force = SensorByName(model, data, "rightfingerforce");
  // std::cout << *left_finger_force << " " << *right_finger_force << std::endl;
  // double pinch_force_target = parameters_[param_counter ++];
  // double finger_force = std::max(std::min(*left_finger_force, -*right_finger_force), 0.0);
  param_counter ++;
  // residual[counter++] = std::max(pinch_force_target - std::min(abs(*left_finger_force), abs(*right_finger_force)), 0.0);

  int finger_touch_obj_id = ReinterpretAsInt(parameters_[param_counter ++]);
  double* left_finger = SensorByName(model, data, "leftfinger");
  double* right_finger = SensorByName(model, data, "rightfinger");
  double* finger_touch_obj = SensorByName(model, data, object_names[finger_touch_obj_id]);
  mju_sub3(residual + counter, left_finger, finger_touch_obj);
  counter += 3;
  mju_sub3(residual + counter, right_finger, finger_touch_obj);
  counter += 3;

  double length1 = std::max(mju_norm3(residual + counter - 6), 0.01);
  double length2 = std::max(mju_norm3(residual + counter - 3), 0.01);
  double dot = mju_dot3(residual + counter - 6, residual + counter - 3);
  // std::cout << length1 << " " << length2 << std::endl;
  double angle_goal = std::max(dot / length1 / length2 + 0.8, 0.0);
  // residual[counter ++] = std::max(dot / length1 / length2 + 0.8, 0.0);
  // std::cout << std::max(0.0, finger_force - 0.08) * 100 << std::endl;
  // std::cout << angle_goal << " " << (length1 + length2) * std::exp(-angle_goal) << std::endl;
  residual[counter ++] = (length1 + length2) * std::exp(-angle_goal - std::max(0.0, 0 - 0.08) * 100) + angle_goal;

  // pinch
  // double* finger_joint = SensorByName(model, data, "fingerjoint");
  // residual[counter ++] = *finger_joint;


  int obj_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int obj_b_id = ReinterpretAsInt(parameters_[param_counter ++]);

  // reach1
  // double* hand = SensorByName(model, data, "hand");
  double* obj_a = SensorByName(model, data, object_names[obj_a_id]);
  // double* box = SensorByName(model, data, "box");
  // double* handle = SensorByName(model, data, "doorhandle");
  double* obj_b = SensorByName(model, data, object_names[obj_b_id]);
  // printf("%d %d\n", object_a_, object_b_);
  double dist = mju_dist3(obj_a, obj_b);
  // mju_sub3(residual + counter, obj_a, obj_b);
  // mju_copy(residual + counter, hand, 3);
  // counter += 3;
  residual[counter++] = std::max(dist - 0.00, 0.0);
  // std::cout << obj_a[0] << " " << obj_a[1] << " " << obj_a[2] << std::endl;

  int obj_2_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int obj_2_b_id = ReinterpretAsInt(parameters_[param_counter ++]);

  // reach2
  // double* hand = SensorByName(model, data, "hand");
  double* obj_2_a = SensorByName(model, data, object_names[obj_2_a_id]);
  // double* box = SensorByName(model, data, "box");
  // double* handle = SensorByName(model, data, "doorhandle");
  double* obj_2_b = SensorByName(model, data, object_names[obj_2_b_id]);
  // double dist2 = mju_dist3(obj_2_a, obj_2_b);
  // printf("%d %d\n", object_a_, object_b_);
  mju_sub3(residual + counter, obj_2_a, obj_2_b);
  // residual[counter + 2] = 0;
  // mju_copy(residual + counter, hand, 3);
  counter += 3;
  // residual[counter++] = std::max(dist2 - 0.00, 0.0);

  // joint
  int joint_id = ReinterpretAsInt(parameters_[param_counter ++]);
  double joint_target = parameters_[param_counter ++];
  double *joint = SensorByName(model, data, joint_names[joint_id]);
  residual[counter++] = std::max(joint_target - abs(*joint), 0.0);
  // printf("%.2f %.2f\n", joint_target, *joint);

  // move away
  int move_obj_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int move_obj_b_id = ReinterpretAsInt(parameters_[param_counter ++]);
  double move_distance_target = parameters_[param_counter ++];
  double *move_obj_a = SensorByName(model, data, object_names[move_obj_a_id]);
  double *move_obj_b = SensorByName(model, data, object_names[move_obj_b_id]);
  // std::cout << joint_id << " " << joint_target << " " << move_obj_a_id << " " << object_names[move_obj_a_id] << " " << move_obj_b_id << " " << object_names[move_obj_b_id] << " " << mju_dist3(move_obj_a, move_obj_b) << std::endl;
  residual[counter++] = std::max(move_distance_target - mju_dist3(move_obj_a, move_obj_b), 0.0);

  // default position
  // double panda_joints_default[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double panda_joints_default[6] = {-0.000378825, -1.57, -0.700116, -0.460413, -2.55436, 0.424858};
  // double panda_joints_default[6] = {-0.000378825, 0.176042, -0.700116, -0.460413, -2.55436, 0.424858};
  // double panda_joints_default[8] = {0.00260707, 0.267844, -0.580238, 0.0102786, -2.53195, 0.149859, 0.373268, -0.189007};
  // double panda_joints_default[8] = {0.00, 0.00 -0.00, 0.0102786, -2.53195, 0.149859, 0.373268, -0.189007};
  // double panda_hand_default[3] = {0.0576433, 0.00168072, 0.579432};
  for (int i = 0; i < 6; i ++) {
    double joint_i = *SensorByName(model, data, "panda_joint" + std::to_string(i));
    // std::cout << joint_i << " ";
    residual[counter ++] = panda_joints_default[i] - joint_i;
  }
  // double* hand = SensorByName(model, data, "hand");
  // mju_sub3(residual + counter, hand, panda_hand_default);
  // counter += 3;

  // std::cout << std::endl;

  // default position no obstruction
  // double panda_joints_default_no_obstruction[8] = {-0.000378825, 0.176042, -0.700116, -0.460413, -2.55436, 0.424858, 0.346362, -0.18275};
  double panda_joints_default_no_obstruction[8] = {3.51141e-07, -0.00079561, -0.343923, -0.000648898, -1.28048, 0.00331369, 0.39396, -0.176195};
  // double panda_joints_default[8] = {0.00260707, 0.267844, -0.580238, 0.0102786, -2.53195, 0.149859, 0.373268, -0.189007};
  // double panda_joints_default[8] = {0.00, 0.00 -0.00, 0.0102786, -2.53195, 0.149859, 0.373268, -0.189007};
  // double panda_hand_default[3] = {0.0576433, 0.00168072, 0.579432};
  for (int i = 0; i < 8; i ++) {
    double joint_i = *SensorByName(model, data, "panda_joint" + std::to_string(i));
    // std::cout << joint_i << " ";
    residual[counter ++] = panda_joints_default_no_obstruction[i] - joint_i;
  }
  // std::cout << std::endl;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

void Kitchen::TransitionLocked(mjModel* model, mjData* data) {
  double residuals[100];
  residual_.Residual(model, data, residuals);
}

}  // namespace panda
}  // namespace mjpc