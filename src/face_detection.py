'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from inference import Network
import time
import numpy as np
import cv2

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
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
        # raise NotImplementedError

    def preprocess_output(self, outputs, image, print_flag = True, threshold = 0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        h, w = image.shape[0:2]
        coords = []
        for item in outputs[0][0]:
            if item[2] >= threshold:
                xmin = int(item[3] * w)
                ymin = int(item[4] * h)
                xmax = int(item[5] * w)
                ymax = int(item[6] * h)
                coords.append((xmin, ymin, xmax, ymax))

                # Drawing the box in the image
                if(print_flag):
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)

        return image, coords        
        # raise NotImplementedError