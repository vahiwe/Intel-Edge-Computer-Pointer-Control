'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
from openvino.inference_engine import IECore

class FacialLandmarkDetectionModel:
    '''
    Class for the Facial Landmark Detection Model.
    '''
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
            output = self.network.requests[0].outputs[self.output_name]
            # reshape output
            output = output.reshape(1, 10)[0]

        # Get the image of the left and right eye
        # Get the coordinates of the eyes from the output
        left_eye_image, right_eye_image, eye_coord = self.preprocess_output(output, face_coord, face)

        # Draw circles around the eyes on the frame
        # Draw the images of the eyes on the frame
        out_frame = self.draw_outputs(output, out_frame, face_coord, left_eye_image, right_eye_image)

        # Return updated image, image of left eye, image of right eye and eye coordinates
        return out_frame, left_eye_image, right_eye_image, eye_coord, inference_time

    def draw_outputs(self, outputs, image, face_coord, left_eye_image, right_eye_image):
        # Create a copy of image
        frame_out = image.copy()

        # height and width of face 
        height = face_coord[3]-face_coord[1] 
        width = face_coord[2]-face_coord[0]

        # draw circles round the eyes
        for i in range(2):
            x = int(outputs[i*2] * width)
            y = int(outputs[i*2+1] * height)
            cv2.circle(frame_out, (face_coord[0]+x, face_coord[1]+y), 30, (0,255,i*255), 2)
        
        # Draw left eye image at the top left hand corner of frame
        frame_out[150:150+left_eye_image.shape[0],20:20+left_eye_image.shape[1]] = left_eye_image

        # Draw right eye image at the top left hand corner of frame
        frame_out[150:150+right_eye_image.shape[0],100:100+right_eye_image.shape[1]] = right_eye_image
        
        # Return updated image
        return frame_out

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

    def preprocess_output(self, outputs, face_coord, face):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # height and width of face 
        height = face_coord[3]-face_coord[1] 
        width = face_coord[2]-face_coord[0]
    
        # Get left and right eye
        left_eye =[outputs[0] * width, outputs[1] * height]
        right_eye = [outputs[2] * width, outputs[3] * height]

        # crop left eye
        left_eye_x_center = left_eye[0]
        left_eye_y_center = left_eye[1]

        # create x,y coordinates for left eye
        left_eye_coord = [left_eye_x_center + face_coord[0], left_eye_y_center + face_coord[1]]

        # check for edges to not crop
        left_eye_y_min = int(left_eye_y_center - 30) if  int(left_eye_y_center - 30) >= 0 else 0 
        left_eye_y_max = int(left_eye_y_center + 30) if  int(left_eye_y_center + 30) <= height else height

        left_eye_x_min = int(left_eye_x_center - 30) if  int(left_eye_x_center - 30) >=0 else 0 
        left_eye_x_max = int(left_eye_x_center + 30) if  int(left_eye_x_center + 30) <= width else width

        # Image of left eye
        left_eye_image = face[left_eye_y_min:left_eye_y_max, left_eye_x_min:left_eye_x_max]

        # crop right eye
        right_eye_x_center = right_eye[0]
        right_eye_y_center = right_eye[1]

        # create x,y coordinates for right eye
        right_eye_coord = [right_eye_x_center + face_coord[0], right_eye_y_center + face_coord[1]]
        
        # check for edges to not crop
        right_eye_y_min = int(right_eye_y_center - 30) if int(right_eye_y_center - 30) >=0 else 0 
        right_eye_y_max = int(right_eye_y_center + 30) if  int(right_eye_y_center + 30) <= height else height

        right_eye_x_min = int(right_eye_x_center - 30) if int(right_eye_x_center - 30) >=0 else 0 
        right_eye_x_max = int(right_eye_x_center + 30) if int(right_eye_x_center + 30) <= width else width

        # Image of left eye
        right_eye_image =  face[right_eye_y_min:right_eye_y_max, right_eye_x_min:right_eye_x_max]

        # create dictionary of eye coordinates
        eye_coord = {"left_eye": left_eye_coord, "right_eye": right_eye_coord}

        # Return left eye image, right eye image and eye coordinates
        return left_eye_image, right_eye_image, eye_coord
