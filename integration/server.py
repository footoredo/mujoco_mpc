from flask import Flask, request, jsonify

app = Flask(__name__)

def dummy_function(data):
    print("Received data:", data)
    return data

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        data = request.json
    else: # for GET request
        data = request.args.to_dict()

    result = dummy_function(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

