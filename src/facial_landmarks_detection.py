'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from inference import Network
import time
import numpy as np
import cv2

class FacialLandmarkDetectionModel:
    '''
    Class for the Facial Landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model = model_name
        self.device = device
        self.network = Network()
        self.extensions = extensions
        # raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.network.load_model(self.model, self.device, self.extensions)
        # raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.network.exec_net(image)
        if self.network.wait() == 0:
            output = (self.network.get_output())[self.network.output_blob]
        return output
        # raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        network_shape = self.network.get_input_shape()
        image_p = np.copy(image)
        image_p = cv2.resize(image_p, (network_shape[3], network_shape[2]))
        image_p = image_p.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
        return image_p

    def preprocess_output(self, outputs, facebox, image, print_flag=True, threshold = 0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        normed_landmarks = outputs.reshape(1, 10)[0]

        height = facebox[3]-facebox[1] 
        width = facebox[2]-facebox[0]
        
        if(print_flag):
            for i in range(2):
                x = int(normed_landmarks[i*2] * width)
                y = int(normed_landmarks[i*2+1] * height)
                cv2.circle(image, (facebox[0]+x, facebox[1]+y), 30, (0,255,i*255), 2)
        
        left_eye_point =[normed_landmarks[0] * width,normed_landmarks[1] * height]
        right_eye_point = [normed_landmarks[2] * width,normed_landmarks[3] * height]
        return image, left_eye_point, right_eye_point
