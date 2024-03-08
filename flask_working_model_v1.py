from fastai.learner import load_learner
from fastai.vision.all import *
from flask import Flask, render_template, request
import os
import base64
from io import BytesIO


image_directory = "test_images"
path = Path(os.getcwd())
full_path = os.path.join(
    path, "D:\Docs\My Projects\Pheumonia app\deploy_image_test\model_kt_v1.pkl")
learner = load_learner(full_path)
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/show-prediction/')
def show_prediction():
    image_file_name = request.args.get("file_name")

    full_path = os.path.join(path, image_directory, image_file_name)
    img = PILImage.create(full_path)
    print(path)
    img_byte_array = BytesIO()
    img.save(img_byte_array, format='PNG')
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
    normal, _, probs = learner.predict(img)
    predict_string = normal
    probs_string = f"{probs[0]:.4f}"
    prediction = {'prediction_key': predict_string}
    probability = {'probability_key': probs_string}
    return (render_template(
        'show-prediction.html',
        prediction=prediction,
        probability=probability,
        img_base64=img_base64))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
