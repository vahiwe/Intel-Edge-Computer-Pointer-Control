# Computer Pointer Controller

This application is a smart IoT solution that uses the Intel® OpenVINO™ software to run inference on a video input and controls the cursor of the PC by using the gaze direction of the users eyes. This application makes use of four different AI models in getting the gaze direction. Below is a diagram showing the inputs and outputs between the models:

![model pipeline](model_pipeline.png)

Below is a video output from the project:

![Giphy of output](Computer_Pointer_Controller.gif)

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

* Logs are available at `logs.txt`.

## Documentation
Below is a documentation of the arguments to the script:

```
usage: main.py [-h] -fdm FACE_DETECTION_MODEL -flm FACIAL_LANDMARK_MODEL -gem
               GAZE_ESTIMATION_MODEL -hpem HEAD_POSE_ESTIMATION_MODEL
               [-d DEVICE] -v VIDEO [-l CPU_EXTENSION] [-pt THRESHOLD]
               [--perf_counts] [--toggle_video] [--visualize_outputs]

optional arguments:
  -h, --help            show this help message and exit
  -fdm FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                        Location of Face Detection Model
  -flm FACIAL_LANDMARK_MODEL, --facial_landmark_model FACIAL_LANDMARK_MODEL
                        Location of Facial Landmark Model
  -gem GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                        Location of Gaze Estimation Model
  -hpem HEAD_POSE_ESTIMATION_MODEL, --head_pose_estimation_model HEAD_POSE_ESTIMATION_MODEL
                        Location of Head Pose Estimation Model
  -d DEVICE, --device DEVICE
                        device to run inference
  -v VIDEO, --video VIDEO
                        Video or Image location. Use 'CAM' as input value to
                        use webcam
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels implementation.
  -pt THRESHOLD, --threshold THRESHOLD
                        Probability threshold for model
  --perf_counts         use the get_perf_counts API to log the time it takes
                        for each layer in the models
  --toggle_video        Allows user toggle video output by pressing Spacebar
                        [Toggle Mode]
  --visualize_outputs   Allows user to visualize model outputs on the frames.
                        It also allows user toggle model output by pressing
                        'o' [Output Toggle Mode]
```

## Directory Structure
Below is the directory structure of the project:
```
├── README.md
├── bin
│   └── demo.mp4
├── main.py
├── model_pipeline.png
├── requirements.txt
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── model.py
    └── mouse_controller.py
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

* Run this command to view line profiling outputs of the application to check for hotspots:

__`❍ python3 -m line_profiler file_name.py.lprof `__

* I used the `get_perf_counts` API to log the time it takes for each layer in the models. You can view `file_name_perf_counts.txt` to analyze more. You can enable it by using the `perf_counts` CLI argument.

* There is a toggle mode that allows you toggle the video output by pressing Spacebar. You can enable it by using the `toggle_video` CLI argument. This increases the total run time of the application.

* There is a toggle mode that allows you toggle the model output by pressing `o`. You can enable it by using the `visualize_outputs` CLI argument. This increases the total run time of the application.

### Edge Cases
Some situations where inference may break are: 
* `PyAutoGUI` fail-safe is triggered from mouse moving to a corner of the screen
* `PyAutoGUI` control of the mouse was too slow, causing the frames to freeze and the application was slow
* Application was crashing if no face was detected
* If lighting is poor then the face might not be detected
* If users face is not looking at the camera in a vertical position then the face might not be detected 

To solve these issues, you have to:
* Disable PyAutoGUI fail-safe. `pyautogui.FAILSAFE = False`
* Reduce precision and speed values of the mousecontroller to improve the speed of the application 
* Add condition to check if face was detected

### References
Here are refences to code sources that was used in this project: 
* https://github.com/vahiwe/Intel_Edge_People_Counter_Project/blob/master/inference.py
* https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
* https://knowledge.udacity.com/questions/171017
* https://knowledge.udacity.com/questions/257811
* https://github.com/baafw/openvino-eye-gaze-estimation
* https://github.com/vahiwe/Intel_Edge_Optimization_Exercises/blob/master/profiling.py
* https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
* https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
* https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
* https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html