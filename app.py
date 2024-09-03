import pickle

from flask import Flask, request, jsonify

model = pickle.load(open('model/model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json(force = True)
    features = data[['feature_0', 'feature_1']]
    prediction = model.predict([features])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)