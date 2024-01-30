from flask import Flask, request, jsonify
from time import sleep
import copy

app = Flask(__name__)

# Dictionaries to store observations and actions
actions = "init" 
observations = "init" 
action_ready = False
obs_ready = False
sleep_time = 0.1


# obs from robot
# action from mpc


@app.route('/obs_ret_act', methods=['POST'])
def obs_ret_act():
    global actions, observations, obs_ready, action_ready, sleep_time
    data = request.json

    observations = copy.deepcopy(data)
    obs_ready = True
    
    while not action_ready:
        sleep(sleep_time)
        
    action_copy = copy.deepcopy(actions)
    action_ready = False
    
    print("====================")
    print(observations)
    print(action_copy)
    return jsonify(action_copy)

@app.route('/act_ret_obs', methods=['POST'])
def act_ret_obs():
    global actions, observations, obs_ready, action_ready, sleep_time
    data = request.json

    # if data["type"] != "init":
    if len(data) > 0:  # waypoints
        actions = copy.deepcopy(data)
        action_ready = True
    
    while not obs_ready:
        sleep(sleep_time)
    
    obs_copy = copy.deepcopy(observations)
    obs_ready = False
    
    print("====================")
    print(obs_copy)
    print(actions)
    return jsonify(obs_copy)

if __name__ == '__main__':
    app.run(debug=True)

