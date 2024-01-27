from flask import Flask, request, jsonify

app = Flask(__name__)

# Dictionaries to store observations and actions
actions = None
observations = None

@app.route('/obs_ret_act', methods=['GET', 'POST'])
def obs_ret_act():
    if request.method == 'POST':
        data = request.json
        observation = data.get('observation')
        return jsonify({"action": action})
    else:
        observation = request.args.get('observation')
        return jsonify({"action": action})

@app.route('/act_ret_obs', methods=['GET', 'POST'])
def act_ret_obs():
    if request.method == 'POST':
        data = request.json
        action = data.get('action')
        return jsonify({"observation": observation})
    else:
        action = request.args.get('action')
        return jsonify({"observation": observation})

if __name__ == '__main__':
    app.run(debug=True)

