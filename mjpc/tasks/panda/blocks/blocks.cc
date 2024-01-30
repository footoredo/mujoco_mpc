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

#include "mjpc/tasks/panda/blocks/blocks.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace panda {

std::string Blocks::XmlPath() const {
  return GetModelPath("panda/blocks/task.xml");
}
std::string Blocks::Name() const { return "Panda Blocks"; }
const std::array<std::string, 5> object_names = {
    "hand", "yellow_block", "red_block", "blue_block", "red_bin"
};

// const std::array<std::string, 2> joint_names = {
//     "leftdoorhinge", "rightdoorhinge"
// };

// ---------- Residuals for in-panda manipulation task ---------
//   Number of residuals: 5
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): cube angular velocity
//     Residual (4): control
// ------------------------------------------------------------
void Blocks::ResidualFn::Residual(const mjModel* model, const mjData* data,
                     double* residual) const {
  int counter = 0;
  int param_counter = 0;

  double *bin_vel = SensorByName(model, data, "bin_vel");
  mju_copy3(residual + counter, bin_vel);
  counter += 3;

  double *red_block_quat = SensorByName(model, data, "red_block_quat");
  double red_block_mat[9] = {0.};
  mju_quat2Mat(red_block_mat, red_block_quat);
  // mju_copy(residual + counter, red_block_quat, 2);
  // std::cout << red_block_quat[0] << " " << red_block_quat[1] << " " << red_block_quat[2] << " " << red_block_quat[3] << std::endl;
  // counter += 2;
  residual[counter ++] = red_block_mat[8] - 1;

  int lift_obj_id = ReinterpretAsInt(parameters_[param_counter ++]);
  double *lift_obj = SensorByName(model, data, object_names[lift_obj_id]);
  double lift_height = parameters_[param_counter ++];
  double obj_height = std::max(0., lift_obj[2] - 0.05);
  residual[counter ++] = std::max(-obj_height + lift_height, 0.);


  int stack_obj_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int stack_obj_b_id = ReinterpretAsInt(parameters_[param_counter ++]);
  double* stack_obj_a = SensorByName(model, data, object_names[stack_obj_a_id]);
  double* stack_obj_b = SensorByName(model, data, object_names[stack_obj_b_id]);

  mju_sub3(residual + counter, stack_obj_a, stack_obj_b);
  residual[counter + 2] -= 0.062;
  counter += 3;


  int obj_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int obj_b_id = ReinterpretAsInt(parameters_[param_counter ++]);

  // std::cout << obj_a_id << " " << obj_b_id << std::endl;

  // reach
  // double* hand = SensorByName(model, data, "hand");
  double* obj_a = SensorByName(model, data, object_names[obj_a_id]);
  // double* box = SensorByName(model, data, "box");
  // double* handle = SensorByName(model, data, "doorhandle");
  double* obj_b = SensorByName(model, data, object_names[obj_b_id]);
  // printf("%d %d\n", object_a_, object_b_);
  // printf("%d %d\n", obj_a_id, obj_b_id);
  mju_sub3(residual + counter, obj_a, obj_b);

  // residual[counter] =  std::tanh((obj_a[0] - obj_b[0])*(obj_a[0] - obj_b[0]) + (obj_a[1] - obj_b[1])*(obj_a[1] - obj_b[1]) + (obj_a[2] - obj_b[2])*(obj_a[2] - obj_b[2]));
  // printf("%.2f\n", residual[counter]);
  // residual[counter + 1] = residual[counter];
  // residual[counter + 2] = residual[counter];
  // printf("%.2f %.2f %.2f\n", obj_a[counter], obj_a[counter + 1], obj_a[2]);
  // printf("%.2f %.2f %.2f\n", obj_b[counter], obj_b[counter + 1], obj_b[2]);
  // mju_copy(residual + counter, hand, 3);
  counter += 3;


  int obj_2_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int obj_2_b_id = ReinterpretAsInt(parameters_[param_counter ++]);

  // std::cout << obj_a_id << " " << obj_b_id << " " << obj_2_a_id << " " << obj_2_b_id << std::endl;

  // reach2
  // double* hand = SensorByName(model, data, "hand");
  double* obj_2_a = SensorByName(model, data, object_names[obj_2_a_id]);
  // double* box = SensorByName(model, data, "box");
  // double* handle = SensorByName(model, data, "doorhandle");
  double* obj_2_b = SensorByName(model, data, object_names[obj_2_b_id]);
  // printf("%d %d\n", object_a_, object_b_);
  
  // residual[counter] = std::tanh((obj_2_a[0] - obj_2_b[0])*(obj_2_a[0] - obj_2_b[0]) + (obj_2_a[1] - obj_2_b[1])*(obj_2_a[1] - obj_2_b[1]) + (obj_2_a[2] - obj_2_b[2])*(obj_2_a[2] - obj_2_b[2]));
  // residual[counter + 1] = residual[counter];
  // residual[counter + 2] = residual[counter];
  mju_sub3(residual + counter, obj_2_a, obj_2_b);
  // mju_copy(residual + counter, hand, 3);
  counter += 3;

  int obj_3_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int obj_3_b_id = ReinterpretAsInt(parameters_[param_counter ++]);

  // std::cout << obj_a_id << " " << obj_b_id << " " << obj_2_a_id << " " << obj_2_b_id << " " << obj_3_a_id << " " << obj_3_b_id << std::endl;

  // reach3
  // double* hand = SensorByName(model, data, "hand");
  double* obj_3_a = SensorByName(model, data, object_names[obj_3_a_id]);
  // double* box = SensorByName(model, data, "box");
  // double* handle = SensorByName(model, data, "doorhandle");
  double* obj_3_b = SensorByName(model, data, object_names[obj_3_b_id]);
  // printf("%d %d\n", object_a_, object_b_);
  // residual[counter] = std::tanh((obj_3_a[0] - obj_3_b[0])*(obj_3_a[0] - obj_3_b[0]) + (obj_3_a[1] - obj_3_b[1])*(obj_3_a[1] - obj_3_b[1]) + (obj_3_a[2] - obj_3_b[2])*(obj_3_a[2] - obj_3_b[2]));
  // residual[counter + 1] = residual[counter];
  //     residual[counter + 2] = residual[counter];
  mju_sub3(residual + counter, obj_3_a, obj_3_b);
  // mju_copy(residual + counter, hand, 3);
  counter += 3;

  // // joint
  // int joint_id = ReinterpretAsInt(parameters_[param_counter ++]);
  // double joint_target = parameters_[param_counter ++];
  // double *joint = SensorByName(model, data, joint_names[joint_id]);
  // residual[counter++] = std::max(joint_target - *joint, 0.0);
  // // printf("%.2f %.2f\n", joint_target, *joint);

  // move away
  int move_obj_a_id = ReinterpretAsInt(parameters_[param_counter ++]);
  int move_obj_b_id = ReinterpretAsInt(parameters_[param_counter ++]);

  // std::cout << move_obj_a_id << " " << move_obj_b_id << " " << obj_a_id << " " << obj_b_id << " " << obj_2_a_id << " " << obj_2_b_id << " " << obj_3_a_id << " " << obj_3_b_id << std::endl;

  double move_distance_target = parameters_[param_counter ++];
  double *move_obj_a = SensorByName(model, data, object_names[move_obj_a_id]);
  double *move_obj_b = SensorByName(model, data, object_names[move_obj_b_id]);


  // mju_sub(residual + counter, move_obj_a, move_obj_b, 2);
  // mju_fill(residual + 2, 0.0, 1);  // only xy
  // std::cout << residual[counter] << " " << residual[counter + 1] << std::endl;
  // counter += 2; // only xy

  residual[counter++] = std::max(move_distance_target - mju_dist3(move_obj_a, move_obj_b), 0.0);
  // residual[counter] = std::max(move_distance_target - mju_norm(residual + counter - 2, 2), 0.0);
  // counter += 1;


  // // bring
  // // double* box1 = SensorByName(model, data, "box1");
  // // double* target1 = SensorByName(model, data, "target1");
  // // mju_sub3(residual + counter, box1, target1);
  // mju_copy(residual + counter, hand, 3);
  // counter += 3;
  // // double* box2 = SensorByName(model, data, "box2");
  // // double* target2 = SensorByName(model, data, "target2");
  // // mju_sub3(residual + counter, box2, target2);
  // mju_copy(residual + counter, hand, 3);
  // counter += 3;

  // open
  // double* doorjoint = SensorByName(model, data, "rightdoorhinge");
  // residual[counter++] = *doorjoint - 1;

  // double* box = SensorByName(model, data, "box");

  // // reach box
  // mju_sub3(residual + counter, hand, box);
  // counter += 3;

  // printf("%.2f %.2f %.2f\n", box[0], box[1], box[2]);

  // move box
  // double target[3] = {1.0, 0.35, 0.4};
  // mju_sub3(residual + counter, box, target);
  // counter += 3;
  // residual[counter ++] = box[0] - 1;

  // actuator
  // std::cout << model->nu << std::endl;
  // mju_copy(residual + counter, data->actuator_force, model->nu);
  // counter += model->nu;

  // end effector to
  double ee_target[3] = {parameters_[param_counter], parameters_[param_counter + 1], parameters_[param_counter + 2]};
  param_counter += 3;
  double* ee_position = SensorByName(model, data, "hand");
  mju_sub3(residual + counter, ee_position, ee_target);
  counter += 3;

  // end effector finger
  // double finger_1 = *SensorByName(model, data, "finger_joint1");
  // double finger_2 = *SensorByName(model, data, "finger_joint2");

  // residual[counter ++] = (0.0407 * 2 - finger_1 - finger_2) * 20;
  // residual[counter ++] = (finger_1 + finger_2) * 20;
  counter += 2;

  // safety
  double *hand_pos = SensorByName(model, data, "hand");
  double finger_vel = *SensorByName(model, data, "finger_joint_jvel");

  residual[counter++] = std::min((hand_pos[2] - 0.03), 0.);
  // counter++;
  // printf("%.2f\n", hand_pos[2]);

  // double *hand_vel = SensorByName(model, data, "hand_vel");
  // double hand_vel_d = (hand_vel[0] * hand_vel[0] + hand_vel[1] * hand_vel[1] + hand_vel[2] * hand_vel[2]);
  // if (hand_vel_d < 10)
  // {
  //   hand_vel_d = 0;
  // }

  // double joint_max = 0;
  // for (int i = 0; i < 7; i++)
  // {
  //   double joint_i = *SensorByName(model, data, "panda_joint" + std::to_string(i+1) + "_jvel");
  //   joint_max = std::max(joint_max, joint_i*joint_i);;
  // }

  // if (joint_max < 10)
  // {
  //   joint_max = 0;
  // } 

  // residual[counter++] = sqrt(hand_vel_d) + sqrt(joint_max);
  if (finger_vel < 0)
  {
	finger_vel = -finger_vel;
  }
  residual[counter++] =  finger_vel;


  // std::cout << finger_1 << " " << finger_2 << std::endl;

  // counter += 2;

  // control cost

  // for (int i = 0; i < model->nu; i++) {
  //   residual[counter ++] = data->actuator_force[i] / model->actuator_gainprm[i * mjNGAIN];
  // }

  // default position
  // double panda_joints_default[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  // double panda_joints_default[8] = {0.000162331, 1.48074, -0.690224, -0.106001, -2.4659, -0.0792418, 1.14184, -1.48384};
  // for (int i = 0; i < 6; i ++) {
  //   double joint_i = *SensorByName(model, data, "panda_joint" + std::to_string(i));
  //   // std::cout << panda_joints[i] << " ";
  //   residual[counter ++] = panda_joints_default[i] - joint_i;
  // }
  counter += 6;

  // default position no obstruction
  // double panda_joints_default_no_obstruction[8] = {-0.00149581, 0.0010889, -0.000380885, -2.96704, -3.06744, -2.9606, 0.342783, 2.96783};
  // double panda_joints_default_no_obstruction[8] = {0.00, 0.00 -0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
  // // double panda_joints_default[8] = {0.000162331, 1.48074, -0.690224, -0.106001, -2.4659, -0.0792418, 1.14184, -1.48384};
  // // double panda_hand_default[3] = {0.0576433, 0.00168072, 0.579432};
  // for (int i = 0; i < 8; i ++) {
  //   double joint_i = *SensorByName(model, data, "panda_joint" + std::to_string(i));
  //   // std::cout << joint_i << " ";
  //   residual[counter ++] = panda_joints_default_no_obstruction[i] - joint_i;
  // }
  counter += 8;
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

void Blocks::TransitionLocked(mjModel* model, mjData* data) {
  double residuals[100];
  // std::cout << 1111 << std::endl;
  residual_.Residual(model, data, residuals);
  // double bring_dist = (mju_norm3(residuals+3) + mju_norm3(residuals+6)) / 2;

  // data->qpos[15] = 1.57;

  // for (int i = 0; i < model->nq; i ++) {
  //   std::cout << data->qpos[i] << ", ";
  // }
  // std::cout << std::endl;

  // reset:
  // if (data->time > 0 && bring_dist < .015) {
  //   // box:
  //   absl::BitGen gen_;
  //   data->qpos[0] = absl::Uniform<double>(gen_, -.5, .5);
  //   data->qpos[1] = absl::Uniform<double>(gen_, -.5, .5);
  //   data->qpos[2] = .05;

  //   // target:
  //   data->mocap_pos[0] = absl::Uniform<double>(gen_, -.5, .5);
  //   data->mocap_pos[1] = absl::Uniform<double>(gen_, -.5, .5);
  //   data->mocap_pos[2] = absl::Uniform<double>(gen_, .03, 1);
  //   data->mocap_quat[0] = absl::Uniform<double>(gen_, -1, 1);
  //   data->mocap_quat[1] = absl::Uniform<double>(gen_, -1, 1);
  //   data->mocap_quat[2] = absl::Uniform<double>(gen_, -1, 1);
  //   data->mocap_quat[3] = absl::Uniform<double>(gen_, -1, 1);
  //   mju_normalize4(data->mocap_quat);
  // }
}

}  // namespace panda
}  // namespace mjpc
