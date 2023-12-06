# Big_data_project-Object_dectection
Project for the Big Data course on object detection for autonomous driving with YOLOv3.

Implemented in Pytorch from scratch [**YOLOv3**](https://arxiv.org/abs/1804.02767) to detect objects relevant for autonomous driving. A simple **web app** has been also created to test the detection model on any driving video given in input (videos from the [calib_challenge](https://github.com/commaai/calib_challenge/tree/main) of commai can be easily retrieved and used) and check the number of frames per second that YOLOv3 can generate.

The backbone network **Darknet-53** (a CNN with 53 convolutional layers) has been trained on a portion of the ImageNet dataset:
- *total images*: 1,281,167
- *classes*: 1,000
- *train set*: 500,000 images (500 per class)
- *test set*: 100,000 (100 per class) 

Due to time constraints, the training has been stopped after the 8th epoch obtaining the following accuracy values:
- **TOP-1**: 0.10123
- **TOP-5**: 0.22513

The **YOLOv3** model has been trained on the [NuImages](https://www.nuscenes.org/nuimages) dataset, which contains 93,000 annotated images and 800,000 2-D bounding boxes for foreground objects, where 36.05% of them are cars and 21.61% are pedestrians. Due to time constraints also the YOLOv3 model has been stopped before finishing the learning phase, at the 28th epoch.

Decent results has been obtained only for the ``car`` class, which will be the only class detected in the web app.

The web app will use CUDA if available. 

On the tested videos, YOLOv3 reached **2 FPS** on CPU and between **31** and **33 FPS** on GPU.

## Run the web app

Before running the web app, download the weights of YOLOv3 from [GoogleDrive](https://drive.google.com/file/d/1Y5Q1WeKqMGleCxYU0HWrqs7pmqSsDxIe/view?usp=sharing) and put the file inside ``docker_web_app/app``.

### Docker
The web app can be run with Docker. Execute inside the folder ``docker_web_app``:
```
docker build -t yolo_web_app .
docker run yolo_web_app
```
Access the web app with the link provided by Flask on terminal.

### Python
The web app can be executed directly with Flask (you will need all the packages in ``requirements.txt`` installed). Execute inside the folder ``docker_web_app/app``:
```
flask --app app run
```
or
```
python3 -m flask --app app run
```
