# Computer Pointer Controller

This application is a smart IoT solution that uses the Intel® OpenVINO™ software to run inference on a video input and controls the cursor of the PC by using the gaze direction of the users eyes. This application makes use of four different AI models in getting the gaze direction. Below is a diagram showing the inputs and outputs between the models:

![model pipeline](model_pipeline.png)

## Project Set Up and Installation
To Setup this project, you need to follow the following steps:

**1)** Firstly download OpenVINO Toolkit and set it up in your local machine according to the instructions [here](https://software.intel.com/en-us/openvino-toolkit/choose-download)

**2)** Fire up your favourite console & clone this repo somewhere:

__`❍ git clone https://github.com/vahiwe/Intel-Edge-Computer-Pointer-Control.git`__

**3)** Enter this directory:

__`❍ cd Intel-Edge-Computer-Pointer-Control `__

**4)** Install [python](https://www.python.org/) if not already installed and run this command to create a virtual environment:

__`❍ python3 -m venv env `__

**5)** Activate the virtual environment:

__`❍ source env/bin/activate `__

**6)** Run this command to install python packages/dependencies:

__`❍ pip install -r requirements.txt `__

**7)** Create a directory for the models:

__`❍ mkdir models/`__

**8)** Download the [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html), [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html), [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html) and [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html). To download the models, we'll make use of the model downloader provided by Intel® OpenVINO™:

* Face Detection Model
```
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 -o models/
```

* Head Pose Estimation Model
```
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 -o models/
```

* Facial Landmarks Detection Model
```
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 -o models/
```

* Gaze Estimation Model
```
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 -o models/
```


## Demo
To run the application, use the below command. You can check the Documentation section for a description of the arguments:

```
python3 main.py -fdm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
        -flm models/intel/landmarks-regression-retail-0009/FP16-Int8/landmarks-regression-retail-0009.xml \
        -hpem models/intel/head-pose-estimation-adas-0001/FP16-Int8/head-pose-estimation-adas-0001.xml \
        -gem models/intel/gaze-estimation-adas-0002/FP16-Int8/gaze-estimation-adas-0002.xml \
        -v bin/demo.mp4 \
        -d CPU
```

## Documentation
Below is a documentation of the arguments to the script:

```
usage: main.py [-h] -fdm FACE_DETECTION_MODEL -flm FACIAL_LANDMARK_MODEL -gem
               GAZE_ESTIMATION_MODEL -hpem HEAD_POSE_ESTIMATION_MODEL
               [-d DEVICE] -v VIDEO [-l CPU_EXTENSION] [-pt THRESHOLD]
               [--no_move] [--no_video]

optional arguments:
  -h, --help            show this help message and exit
  -fdm FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                        location of face detection model model to be used
  -flm FACIAL_LANDMARK_MODEL, --facial_landmark_model FACIAL_LANDMARK_MODEL
                        location of facial landmark model to be used
  -gem GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                        location of gaze estimation model to be used
  -hpem HEAD_POSE_ESTIMATION_MODEL, --head_pose_estimation_model HEAD_POSE_ESTIMATION_MODEL
                        location of head pose estimation model to be used
  -d DEVICE, --device DEVICE
                        device to run inference
  -v VIDEO, --video VIDEO
                        video location
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -pt THRESHOLD, --threshold THRESHOLD
                        Probability threshold for model
  --no_move             Not move mouse based on gaze estimation output
  --no_video            Don't show video window
```

## Benchmarks
The table below shows the load time and inference time for the different models on different precisions. The Face Detection model only has one precision (FP32).

| Factor/Model       | Face Detection   | Landmarks Detetion        | Headpose Estimation | Gaze Estimation |
|--------------------|---------------|-----------|-------------|-----------|
|Load Time FP32      |  327ms        | 121ms     | 135ms       | 145ms     |
|Load Time FP16      |  NA           | 122ms     | 125ms       | 149ms     |  
|Load Time FP16-INT8 |  NA           | 185ms     | 260ms       | 287ms     |
||||||
|Inference Time FP32 | 12.1ms         | 0.5ms     | 1.2ms       | 1.5ms     |
|Inference Time FP16 | NA            | 0.5ms     | 1.2ms       | 1.5ms     |
|Inference Time FP16-INT8| NA        | 0.5ms     | 1.0ms       | 1.1ms |
||||||


## Results
* Load time for models with FP32 is less than FP16 and the same for FP16 models is less than INT8. 
* Inference time is the same across all model precisions.

## Stand Out Suggestions
For further improvement of the application, one can use [Deep Learning Workbench](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Package.html) to get model performance summary, and [VTune Amplifier](https://software.intel.com/content/www/us/en/develop/tools/vtune-profiler.html) to measure hot spots in the application code

### Edge Cases
Some situations where inference may break are: 
* `PyAutoGUI` fail-safe is triggered from mouse moving to a corner of the screen
* `PyAutoGUI` control of the mouse was too slow, causing the frames to freeze and the application was slow
* Application was crashing if no face was detected

To solve these issues, you have to:
* Disable PyAutoGUI fail-safe. `pyautogui.FAILSAFE = False`
* Reduce precision and speed values of the mousecontroller to improve the speed of the application 
* Add condition to check if face was detected