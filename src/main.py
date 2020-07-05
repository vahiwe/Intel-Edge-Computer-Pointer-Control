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
import sys
import time
import socket
import json
import cv2

import logging


from argparse import ArgumentParser
from face_detection import FaceDetectionModel
from input_feeder import InputFeeder
import sys
import numpy as np
from random import randint
from inference import Network
from facial_landmarks_detection import FacialLandmarkDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel
from mouse_controller import MouseController

#import _thread



def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--fdmodel", required=True, type=str,
                        help="Path to a face detection xml file with a trained model.")
    parser.add_argument("-hpm", "--hpmodel", required=True, type=str,
                        help="Path to a head pose estimation xml file with a trained model.")
    parser.add_argument("-lmm", "--lmmodel", required=True, type=str,
                        help="Path to a facial landmarks xml file with a trained model.")
    parser.add_argument("-gem", "--gemodel", required=True, type=str,
                        help="Path to a gaze estimation xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path video file or CAM to use camera")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    
    parser.add_argument("--print",default=False,
                        help="Print models output on frame",action="store_true")
    
    parser.add_argument("--no_move",default=False,
                        help="Not move mouse based on gaze estimation output",action="store_true")
    
    parser.add_argument("--no_video",default=False,
                        help="Don't show video window",action="store_true")

    return parser


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    # Initialise the mouse controller class
    mc = MouseController("low","slow")
    fdnet = FaceDetectionModel(args.fdmodel)
    lmnet = FacialLandmarkDetectionModel(args.lmmodel)
    hpnet = HeadPoseEstimationModel(args.hpmodel)
    genet = GazeEstimationModel(args.gemodel)
    video_file = args.input

    ### Load the model ###
    print("============== Models Load time ===============") 
    start_time = time.time()
    fdnet.load_model()
    print("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    lmnet.load_model()
    print("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    hpnet.load_model()
    print("Headpose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    genet.load_model()
    print("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start_time)) )
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
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    counter=0
    fd_infertime = 0
    lm_infertime = 0
    hp_infertime = 0
    ge_infertime = 0
    start_inference_time=time.time()

    try:
        #loop until stream is over
        while cap.isOpened:
            #  Read from the video capture ###
            flag, frame = cap.read()
            if not flag:
                break

            key_pressed = cv2.waitKey(60)
            counter += 1
            
            # face detection
            p_frame = fdnet.preprocess_input(frame)
            start_time = time.time()
            fnoutput = fdnet.predict(p_frame)
            fd_infertime += time.time() - start_time
            out_frame,fboxes = fdnet.preprocess_output(fnoutput,frame,args.print)
            
            #for each face
            for fbox in fboxes:
                
                # get face landmarks
                # crop face from frame
                face = frame[fbox[1]:fbox[3],fbox[0]:fbox[2]]
                p_frame = lmnet.preprocess_input(face)
                
                start_time = time.time()
                lmoutput = lmnet.predict(p_frame)
                lm_infertime += time.time() - start_time
                out_frame,left_eye_point,right_eye_point = lmnet.preprocess_output(lmoutput, fbox, out_frame,args.print)

                # get head pose estimation
                p_frame  = hpnet.preprocess_input(face)
                start_time = time.time()
                hpoutput = hpnet.predict(p_frame)
                hp_infertime += time.time() - start_time
                out_frame, headpose_angels = hpnet.preprocess_output(hpoutput,out_frame, face,fbox,args.print)

                # get gaze  estimation
                out_frame, left_eye, right_eye  = genet.preprocess_input(out_frame,face,left_eye_point,right_eye_point,args.print)
                start_time = time.time()
                geoutput = genet.predict(left_eye, right_eye, headpose_angels)
                ge_infertime += time.time() - start_time
                out_frame, gazevector = genet.preprocess_output(geoutput,out_frame,fbox, left_eye_point,right_eye_point,args.print)

                if(not args.no_video):
                    cv2.imshow('im', out_frame)
                
                if(not args.no_move):
                    mc.move(gazevector[0],gazevector[1])
                
                #consider only first detected face in the frame
                break
            
            # Break if escape key pressed
            if key_pressed == 27:
                break

        #logging inference times
        if(counter>0):
            print("============== Models Inference time ===============") 
            print("Face Detection:{:.1f}ms".format(1000* fd_infertime/counter))
            print("Facial Landmarks Detection:{:.1f}ms".format(1000* lm_infertime/counter))
            print("Headpose Estimation:{:.1f}ms".format(1000* hp_infertime/counter))
            print("Gaze Estimation:{:.1f}ms".format(1000* ge_infertime/counter))
            print("============== End ===============================") 

        # Release the capture and destroy any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run inference: ", e)
        

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)
    

if __name__ == '__main__':
    main()