'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import math
import numpy as np
from openvino.inference_engine import IECore


class HeadPoseEstimationModel:
    '''
    Class for the Head Pose Estimation Model.
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
        self.outputs = iter(self.model.outputs)
        self.first_output_name = next(self.outputs)
        self.second_output_name = next(self.outputs)
        self.third_output_name = next(self.outputs)
        self.output_shape = self.model.outputs[self.first_output_name].shape

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
    def predict(self, image, face_coord, out_frame):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # Get Image of face from the frame using the output from face detection model
        face = image[face_coord[1]:face_coord[3],face_coord[0]:face_coord[2]]

        # Preprocess input by resizing it
        frame = self.preprocess_input(face)

        start_time = time.time()
        # Run inference on the processed input
        self.network.start_async(request_id=0, inputs={self.input_name: frame})
        
        # get inference time
        inference_time = time.time() - start_time

        # Retrieve output from the inference engine
        if self.network.requests[0].wait(-1) == 0:
            output = self.network.requests[0].outputs
            
        # Get the head pose angles from the output
        angles = self.preprocess_output(output)

        # Write the angles on the frame and visualize the angles
        out_frame = self.draw_outputs(angles, out_frame, face_coord)

        # Return updated image and head pose angles 
        return out_frame, angles, inference_time

    # code source: https://knowledge.udacity.com/questions/171017
    def draw_outputs(self, angles, image, face_coord):
        # Create a copy of image
        frame_out = image.copy()

        # write the yaw, pitch and roll to the frame as text
        cv2.putText(frame_out,"yaw:{:.1f}".format(angles[0]), (20,20), 0, 0.6, (255,255,0))
        cv2.putText(frame_out,"pitch:{:.1f}".format(angles[1]), (20,40), 0, 0.6, (255,255,0))
        cv2.putText(frame_out,"roll:{:.1f}".format(angles[2]), (20,60), 0, 0.6, (255,255,0))
        
        # visualize head pose 
        xmin, ymin, xmax , ymax = face_coord
        height = xmax - xmin 
        width = ymax - ymin
        face_center = (xmin + height / 2, ymin + width / 2, 0)
        self.draw_axes(frame_out, face_center, angles[0], angles[1], angles[2])

        # Return updated image
        return frame_out

    # code source: https://knowledge.udacity.com/questions/171017
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll):
        focal_length = 950.0
        scale = 50

        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])
        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame

    # code source: https://knowledge.udacity.com/questions/171017
    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix
    
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
    # code source: https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # Resize image to fit input requirements of the model
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
        return image_p

    # code source: https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # Retrieve head pose angles (yaw, pitch and roll) from the output
        angles = []
        angles.append(outputs['angle_y_fc'][0][0])
        angles.append(outputs['angle_p_fc'][0][0])
        angles.append(outputs['angle_r_fc'][0][0])
        
        # Return angles
        return angles
    