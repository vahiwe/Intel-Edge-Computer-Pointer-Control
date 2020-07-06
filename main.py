"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import time
import cv2
from argparse import ArgumentParser
from src.face_detection import FaceDetectionModel
from src.facial_landmarks_detection import FacialLandmarkDetectionModel
from src.head_pose_estimation import HeadPoseEstimationModel
from src.gaze_estimation import GazeEstimationModel
from src.mouse_controller import MouseController

def main(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    """
    # Initialise the mouse controller class
    mc = MouseController("higher", "faster")

    # Initialise models instance and video input
    face_detection_model = FaceDetectionModel(args.face_detection_model)
    facial_landmark_model = FacialLandmarkDetectionModel(args.facial_landmark_model)
    head_pose_model = HeadPoseEstimationModel(args.head_pose_estimation_model)
    gaze_estimation_model = GazeEstimationModel(args.gaze_estimation_model)
    video_file = args.video

    ### Load the models ###

    # Start time for loading models
    model_load_start_time = time.time()

    print("============== Models Load time ===============") 

    # Calculate model load time for face detection model
    start_time = time.time()
    face_detection_model.load_model()
    print("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    # Calculate model load time for facial landmark detection model
    start_time = time.time()
    facial_landmark_model.load_model()
    print("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    # Calculate model load time for head pose model
    start_time = time.time()
    head_pose_model.load_model()
    print("Headpose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    # Calculate model load time for gaze estimation model
    start_time = time.time()
    gaze_estimation_model.load_model()
    print("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    # Calculate total models load time
    print("Total model load time: {:.1f}ms".format(1000 * (time.time() - model_load_start_time)) )
    
    print("==============  End =====================") 
    
    # Checks for live feed
    if video_file == 'CAM':
        input_stream = 0
        single_image_mode = False
    # Checks for input image
    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :
        single_image_mode = True
        input_stream = video_file
    # Checks for video file
    else:
        input_stream = video_file
        assert os.path.isfile(video_file), "Specified input file doesn't exist"
        single_image_mode = False

    try:
        cap=cv2.VideoCapture(input_stream)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    if input_stream:
        cap.open(input_stream)

    no_of_frames=0
    face_detectiom_inference_time = 0
    facial_landmark_inference_time = 0
    head_pose_inference_time = 0
    gaze_estimation_inference_time = 0
    start_inference_time = time.time()

    try:
        # loop until stream is over
        while cap.isOpened:
            # Read from the video capture
            flag, frame = cap.read()
            if not flag:
                break

            key_pressed = cv2.waitKey(60)
            no_of_frames += 1
            
            # face detection
            face_coord, out_frame, inference_time = face_detection_model.predict(frame) 

            # Check if face was detected or not
            if len(face_coord) == 0:
                continue  
            
            # calculate total inference time of face detection model
            face_detectiom_inference_time += inference_time

            # get face landmarks
            out_frame,left_eye_image,right_eye_image, eye_coord, inference_time = facial_landmark_model.predict(frame, face_coord, out_frame)
            
            # calculate total inference time of facial landmarks model
            facial_landmark_inference_time += inference_time

            # get head pose estimation
            out_frame, headpose_angles, inference_time = head_pose_model.predict(frame, face_coord, out_frame)
            
            # calculate total inference time of head pose estimation model
            head_pose_inference_time += inference_time

            # get gaze estimation
            out_frame, gaze_vector, inference_time  = gaze_estimation_model.predict(left_eye_image, right_eye_image, headpose_angles, eye_coord, out_frame)
            
            # calculate total inference time of gaze estimation model
            gaze_estimation_inference_time += inference_time

            if(not args.no_video):
                if single_image_mode:
                    cv2.imwrite('output_image.jpg', out_frame)
                else:
                    cv2.imshow('Mouse Pointer Control', out_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            if(not args.no_move):
                mc.move(gaze_vector[0],gaze_vector[1])
        
            # Break if escape key pressed
            if key_pressed == 27:
                break


        print("\n Number of frames: {} \n".format(no_of_frames))
        print("============== Inference Time ===============") 
        print("Face Detection Inference Time:{:.1f}ms".format(1000* face_detectiom_inference_time/no_of_frames))
        print("Facial Landmarks Detection Inference Time:{:.1f}ms".format(1000* facial_landmark_inference_time/no_of_frames))
        print("Headpose Estimation Inference Time:{:.1f}ms".format(1000* head_pose_inference_time/no_of_frames))
        print("Gaze Estimation Inference Time:{:.1f}ms".format(1000* gaze_estimation_inference_time/no_of_frames))
        print("Total time:{:.1f}s".format(round((time.time() - start_inference_time), 1)))
        print("============== End ===============================") 

        # Release the capture and destroy any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run inference: ", e)
    

if __name__ == '__main__':
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("-fdm", '--face_detection_model', type=str, required=True, default="models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001", help="location of face detection model model to be used")
    parser.add_argument("-flm", '--facial_landmark_model', type=str, required=True, default="models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009", help="location of facial landmark model to be used")
    parser.add_argument("-gem", '--gaze_estimation_model', type=str, required=True, default="models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002", help="location of gaze estimation model to be used")
    parser.add_argument("-hpem", '--head_pose_estimation_model', type=str, required=True, default="models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001", help="location of head pose estimation model to be used")
    parser.add_argument("-d", '--device', type=str, default='CPU', help="device to run inference")
    parser.add_argument("-v", '--video',type=str, required=True, default="bin/demo.mp4", help="video location")
    parser.add_argument("-l", '--cpu_extension', type=str, required=False, default=None, help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-pt", '--threshold', type=float, default=0.60, help="Probability threshold for model")
    parser.add_argument("--no_move",default=False,
                        help="Not move mouse based on gaze estimation output",action="store_true")
    parser.add_argument("--no_video",default=False,
                        help="Don't show video window",action="store_true")

    args=parser.parse_args()

    # Run main function
    main(args)