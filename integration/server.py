from flask import Flask, request, jsonify
from time import sleep

app = Flask(__name__)

# Dictionaries to store observations and actions
actions = "init" 
observations = "init" 
obs_turn = True
sleep_time = 0.5

@app.route('/obs_ret_act', methods=['POST'])
def obs_ret_act():
    global actions, observations, obs_turn, sleep_time
    data = request.json

    while not obs_turn:
        sleep(sleep_time)
    obs_turn = False

    observations = data.get('observations')
    print("====================")
    print(observations)
    print(actions)
    return jsonify({"actions": actions})

@app.route('/act_ret_obs', methods=['POST'])
def act_ret_obs():
    global actions, observations, obs_turn, sleep_time
    data = request.json

    while obs_turn:
        sleep(sleep_time)
    obs_turn = True

    actions = data.get('actions')
    print("====================")
    print(observations)
    print(actions)
    return jsonify({"observations": observations})

if __name__ == '__main__':
    app.run(debug=True)

