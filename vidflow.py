import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow.processors.vision.detectors import TensorflowObjectDetector
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self):
        super(FrameIndexSplitter, self).__init__()
    
    def process(self, data):
        index, frame = data
        return frame

input_file =  "input.mp4"
output_file = "output.avi"
reader = VideofileReader(input_file)
frame = FrameIndexSplitter()(reader)
detector = TensorflowObjectDetector()(frame)
annotator = BoundingBoxAnnotator()(frame, detector)
writer = VideofileWriter(output_file, fps = 30)(annotator)
fl = flow.Flow([reader], [writer], flow_type = BATCH)
fl.run()
fl.join()
