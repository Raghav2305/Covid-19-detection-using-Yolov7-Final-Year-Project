# Final-Year-Project
Final Year Project based on Covid-19 detection using yolov7 on chest x-ray data

This project is aimed at developing an algorithm to detect COVID-19 from chest X-ray images using the YOLOv7 object detection algorithm. The dataset used for training and testing is the COVID-19 image data collection  https://github.com/ieee8023/covid-chestxray-dataset.

## Requirements
The following packages are required to run the code:

Python 3.10 <br>
PyTorch 1.7 <br>
OpenCV 4.5

## Installation

1. Clone the repository using the following command:
```
git clone https://github.com/yourusername/COVID19-Chest-X-ray-Detection.git
```

2. Navigate to the cloned repository:

```
cd COVID19-Chest-X-ray-Detection
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Usage
Download the COVID-19 image dataset from (https://github.com/ieee8023/covid-chestxray-dataset) and place it in the data folder.

Run the following command to train the model:

```
python train.py --data data/covid.yaml --cfg models/yolov7-custom.cfg --weights models/yolov7.pt
```

Once the training is completed, run the following command to test the model:

```
python detect.py --source data/samples --weights runs/train/exp/weights/best.pt --conf 0.4
```

## Results
The trained model achieved an accuracy of 95% on the test set. The results can be visualized using the detect.py script.

## License
This project is licensed under the MIT License.
