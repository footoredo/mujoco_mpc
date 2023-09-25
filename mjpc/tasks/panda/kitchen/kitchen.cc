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

const std::array<std::string, 10> object_names = {
	"hand", "cabinet_doorhandle_r", "cabinet_doorhandle_l", "kettle_handle", "microwave_handle", "slide_handle", "knob1", "knob2", "knob3", "knob4"
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


  int obj_a_id = ReinterpretAsInt(parameters_[param_counter ++]);

  int obj_b_id = ReinterpretAsInt(parameters_[param_counter ++]);


  // reach

  // double* hand = SensorByName(model, data, "hand");

  double* obj_a = SensorByName(model, data, object_names[obj_a_id]);

  // double* box = SensorByName(model, data, "box");

  // double* handle = SensorByName(model, data, "doorhandle");

  double* obj_b = SensorByName(model, data, object_names[obj_b_id]);

  // printf("%d %d\n", object_a_, object_b_);

  mju_sub3(residual + counter, obj_a, obj_b);

  // mju_copy(residual + counter, hand, 3);

  counter += 3;


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

  // std::cout << joint_id << " " << joint_target << " " << move_obj_a_id << " " << object_names[move_obj_a_id] << " " << move_obj_b_id << " " << object_names[move_obj_b_id] << std::endl;

  residual[counter++] = std::max(move_distance_target - mju_dist3(move_obj_a, move_obj_b), 0.0);


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