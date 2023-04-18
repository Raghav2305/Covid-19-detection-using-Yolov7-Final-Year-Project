import torch
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights_path = "./Yolov7/best.pt"
# Load the best weights from file
model = torch.hub.load("WongKinYiu/yolov7", "custom",
                       weights_path, trust_repo=True)
model.to(device)
# Set the model in evaluation mode
# model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prepare the input image
image = cv2.imread('./test_on_these_images/Covid_1.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image)
class_labels = results.pandas().xyxy[0]['name'].tolist()
print(class_labels)
# input_image = transform(image).unsqueeze(0).to(device)


# # Run the model on the input image
# with torch.no_grad():
#     outputs = model(input_image)

# # Print the predicted bounding boxes and classes
# print(outputs.xyxy[0])
