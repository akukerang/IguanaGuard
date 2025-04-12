import modules.camera as cam
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType) 
import numpy as np

class Detection_Model:
    def __init__(self, confidence=0.65):
        #=== Hailo Setup ===
        self.target = VDevice()
        self.CONFIDENCE_THRESHOLD = confidence
        self.hef_path = 'models/yolov11m.hef'
        self.hef = HEF(self.hef_path)

        # Configure network groups
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        # Create input and output virtual streams params
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        # Define dataset params
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape

        self.undistort_frame = ""
        self.center_x = 0
        self.center_y = 0
        self.detected = False

    
    def run(self, frame):
        self.undistorted_frame = cam.undistort_frame(frame) # Remove fisheye effect        
        preprocessed = cam.preprocess(self.undistorted_frame)
        dim = 640
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(preprocessed)  
                oh, ow, _ = self.undistorted_frame.shape
                rh, rw = oh / dim, ow / dim
                detections = np.array(infer_results[list(infer_results.keys())[0]][0])
                self.detected = False
                if detections[0].shape[0] > 0: # Iguana detected in frame
                    flattened_detections = detections[0].reshape(-1, 5)
                    sorted_detections = flattened_detections[flattened_detections[:, 4].argsort()[::-1]] # Gets box with highest confidence
                    highest_conf = sorted_detections[0]
                    if highest_conf[4] > self.CONFIDENCE_THRESHOLD:
                        self.detected = True 
                        bbox = highest_conf[0:4] * dim
                        self.center_x, self.center_y = cam.draw_detection(self.undistorted_frame, bbox, highest_conf[4], (0,255,0), rw, rh) # Draws bounding boxes
