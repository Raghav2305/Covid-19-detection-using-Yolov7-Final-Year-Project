from flask import Flask, url_for, redirect, request
from flask import render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
import cv2

app = Flask(__name__, template_folder='templates')
# model = load_model("./Resnet50/resnet50.h5")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = "./Yolov7/best.pt"
# Load the best weights from file
model = torch.hub.load("WongKinYiu/yolov7", "custom",
                       weights_path, trust_repo=True)
model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image)
    predictions = results.pandas().xyxy[0]['name'].tolist()

    # print(image)
    # # preprocess the image
    # image = np.array(image.resize((224, 224))) / 255.0
    # # make predictions
    # predictions = model.predict(np.array([image]))
    # # get the predicted class


# print the predicted class (assuming binary classification)
    # class_idx = np.argmax(predictions)
    return redirect(url_for('predict', class_idx=predictions[0]))


@app.route("/predict/<string:class_idx>")
def predict(class_idx):
    if class_idx == "Covid":
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
