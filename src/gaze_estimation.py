'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import math
from openvino.inference_engine import IECore

class GazeEstimationModel:
    '''
    Class for the Gaze Estimation Model.
    '''
    # code source: https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = os.path.splitext(model_name)[0] + ".bin"
        self.model_structure = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None

        try:
            self.model = IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    # code source: https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.check_model()
        self.network = self.plugin.load_network(self.model, self.device)
        return

    # code source: https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
    def predict(self, left_eye_image, right_eye_image, headpose_angles, eye_coord, out_frame):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # Preprocess input by resizing it
        left_eye_frame, right_eye_frame = self.preprocess_input(left_eye_image, right_eye_image)
        
        start_time = time.time()
        # Run inference on the processed input
        self.network.start_async(request_id=0, inputs={'left_eye_image':left_eye_frame, 'right_eye_image':right_eye_frame, 'head_pose_angles':headpose_angles})
        
        # get inference time
        inference_time = time.time() - start_time

        # Retrieve output from the inference engine
        if self.network.requests[0].wait(-1) == 0:
            output = self.network.requests[0].outputs[self.output_name]

        # Get the mouse coordinates and gaze vextor from the output
        gaze_vector = self.preprocess_output(output)
        
        # Draw an arrow showing the gaze direction on the frame
        out_frame = self.draw_outputs(gaze_vector, out_frame, eye_coord)

        # Return updated image, mouse coordinates and gaze vector
        return out_frame, gaze_vector, inference_time
    
    # code source: https://knowledge.udacity.com/questions/257811
    # code source: https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
    def draw_outputs(self, gaze_vector, image, eye_coord):
        # Create a copy of image
        frame_out = image.copy()

        # Write the gaze vextor values to the frame
        cv2.putText(frame_out,"x:"+str('{:.1f}'.format(gaze_vector[0]*100))+",y:"+str('{:.1f}'.format(gaze_vector[1]*100))+",z:"+str('{:.1f}'.format(gaze_vector[2])) , (20, 100), 0,0.6, (0,0,255), 1)

        # Draw an arrow showing the gaze direction for the two eyes
        cv2.arrowedLine(frame_out, (int(eye_coord['left_eye'][0]),int(eye_coord['left_eye'][1])), (int(eye_coord['left_eye'][0]) + int(gaze_vector[0]*100), int(eye_coord['left_eye'][1]) + int(-gaze_vector[1]*100)), (255, 100, 100), 5)
        cv2.arrowedLine(frame_out, (int(eye_coord['right_eye'][0]),int(eye_coord['right_eye'][1])), (int(eye_coord['right_eye'][0]) + int(gaze_vector[0]*100),int(eye_coord['right_eye'][1]) + int(-gaze_vector[1]*100)), (255,100, 100), 5)
        
        # Return updated image
        return frame_out

    # code source: https://github.com/vahiwe/Intel_Edge_People_Counter_Project/blob/master/inference.py
    def check_model(self):
        ### TODO check if all layers are supported
        ### return True if all supported, False otherwise
        layers_supported = self.plugin.query_network(self.model, device_name='CPU')
        layers = self.model.layers.keys()

        all_supported = True
        for l in layers:
            if l not in layers_supported:
                all_supported = False

        if not all_supported:
            ### TODO: Add any necessary extensions ###
            self.plugin.add_extension(self.extensions, self.device)

    # code source: https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
    def preprocess_input(self, left_eye_image, right_eye_image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # Resize image to fit input requirements of the model
        left_eye_frame = cv2.resize(left_eye_image, (60, 60))
        left_eye_frame = left_eye_frame.transpose((2,0,1))
        left_eye_frame = left_eye_frame.reshape(1, *left_eye_frame.shape)

        right_eye_frame = cv2.resize(right_eye_image, (60, 60))
        right_eye_frame = right_eye_frame.transpose((2,0,1))
        right_eye_frame = right_eye_frame.reshape(1, *right_eye_frame.shape)

        return left_eye_frame, right_eye_frame

    # code source: https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # get gaze vector and mouse coordinates from output
        gaze_vector = outputs[0]

        # Return mouse coordinates and gaze vector
        return gaze_vector
