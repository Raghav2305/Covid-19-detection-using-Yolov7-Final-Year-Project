from flask import Flask, url_for, redirect, request
from flask import render_template
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__, template_folder='templates')
model = load_model("./Resnet50/resnet50.h5")


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/compare")
def compare():
    return render_template("comparator.html")


@app.route("/impact")
def impact():
    return render_template("impact.html")


@app.route("/upload_files", methods=['POST'])
def upload_files():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    print(file)
    if file.filename == '':
        return 'No selected file'

    image = Image.open(file.stream)
    print(image)
    # preprocess the image
    image = np.array(image.resize((224, 224))) / 255.0
    # make predictions
    predictions = model.predict(np.array([image]))
    # get the predicted class


# print the predicted class (assuming binary classification)
    class_idx = np.argmax(predictions)
    return redirect(url_for('predict', class_idx=class_idx))


@app.route("/predict/<int:class_idx>")
def predict(class_idx):
    if class_idx == 0:
        prediction = "COVID"
    else:
        prediction = "NORMAL"
    return render_template('predict.html', prediction=prediction)

    # class_idx = request.args.get('class_idx')
    # if(class_idx == 0):
    #     return "Covid"
    # else:
    #     return "Normal"


if __name__ == "__main__":
    app.run(debug=True)
