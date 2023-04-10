import cv2
import torch
import numpy as np

# Load the PyTorch model
model = torch.load("model.pt")

# Load the input image
img = cv2.imread("input_image.jpg")

# Preprocess the image
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Make a prediction using the model
with torch.no_grad():
    output = model(torch.tensor(img))

# Get the predicted class label
pred_label = np.argmax(output.numpy())

# Print the predicted class label
print("Predicted class label: ", pred_label)
