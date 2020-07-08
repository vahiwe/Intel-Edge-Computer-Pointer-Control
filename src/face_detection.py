'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import atexit
import line_profiler
from openvino.inference_engine import IECore
# code source: https://github.com/vahiwe/Intel_Edge_Optimization_Exercises/blob/master/profiling.py
profile=line_profiler.LineProfiler()
# this prints the profiling stats to sys.stdout
# atexit.register(profile.print_stats)


# this saves the profiling stats to a file
atexit.register(profile.dump_stats, "face_detection.py.lprof")

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    # code source: https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
    def __init__(self, model_name, threshold = 0.5, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = os.path.splitext(model_name)[0] + ".bin"
        self.model_structure = model_name
        self.device = device
        self.extensions = extensions
        self.threshold = threshold
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
    @profile
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
    @profile
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # Preprocess input by resizing it
        frame = self.preprocess_input(image)

        start_time = time.time()
        # Run inference on the processed input
        self.network.start_async(request_id=0, inputs={self.input_name: frame})

        # get inference time
        inference_time = time.time() - start_time

        # Retrieve output from the inference engine
        if self.network.requests[0].wait(-1) == 0:
            output = self.network.requests[0].outputs[self.output_name]

        # Get the coordinates of the face from the output
        face_coords = self.preprocess_output(output, image)

        # Add condition to check if face was detected
        if len(face_coords) > 0:
            # Select the first face from the outputs 
            face_coord = face_coords[0]

            # Draw a bounding box around the face on the frame
            image = self.draw_outputs(face_coord, image)
        else:
            face_coord = []

        # Return face coordinates and updated image 
        return face_coord, image, inference_time

    # code source: https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
    @profile
    def draw_outputs(self, face_coord, image):
        # Create a copy of image
        frame_out = image.copy()

        # Draw bounding boxes around all detected faces
        cv2.rectangle(frame_out, face_coord[:2], face_coord[2:], (0, 255, 0), 2)

        # Return updated image
        return frame_out

    # code source: https://github.com/vahiwe/Intel_Edge_People_Counter_Project/blob/master/inference.py
    @profile
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
    # code source: https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
    @profile
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

    # code source: https://github.com/vahiwe/Intel_Edge_Smart_Queuing_System/blob/master/Create_Python_Script.ipynb
    # code source: https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
    @profile
    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # Retrieve coordinates of faces present in the output
        h, w = image.shape[0:2]
        coords = []
        for item in outputs[0][0]:
            if item[2] >= self.threshold:
                xmin = int(item[3] * w)
                ymin = int(item[4] * h)
                xmax = int(item[5] * w)
                ymax = int(item[6] * h)
                coords.append((xmin, ymin, xmax, ymax))

        # Return coordinates
        return coords