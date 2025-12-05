import os
import torch
import numpy as np
from options.test_options import TestOptions
from models import create_model
from util import util
import cv2
from datetime import datetime

# Define hard-coded parameters and paths
video_path = r"sd_video_2.mp4"  # Path to the input video
results_dir = './results/'            # Directory to save results
model_name = 'desmoke'  # Model name
model_type = 'test'                   # Model type
no_dropout = True                     # No dropout flag

if __name__ == '__main__':
    # Create TestOptions object and set parameters
    opt = TestOptions().parse()
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.results_dir = results_dir  # Set results directory
    opt.name = model_name   # Set model name
    opt.model = model_type  # Set model type
    if no_dropout:
        opt.no_dropout = True  # Set no dropout flag
    else:
        opt.no_dropout = False

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_video_path = os.path.join(results_dir, f'output_video_{timestamp}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        input_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame_rgb = np.asarray([input_frame_rgb])
        input_frame_rgb = np.transpose(input_frame_rgb, (0, 3, 1, 2))
        data = {"A": torch.FloatTensor(input_frame_rgb), "A_paths": [video_path]}

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        result_frame = model.get_current_visuals()['fake']
        result_frame = util.tensor2im(result_frame)

        # Correct color channel: convert the result to BGR properly
        result_frame_bgr = cv2.cvtColor(np.array(result_frame), cv2.COLOR_RGB2BGR)

        # Resize result frame to match the original frame dimensions
        if result_frame_bgr.shape[:2] != frame.shape[:2]:
            result_frame_bgr = cv2.resize(result_frame_bgr, (frame.shape[1], frame.shape[0]))

        # Fuse the input frame with the result frame side-by-side
        fused_frame = np.hstack((frame, result_frame_bgr))

        # Write the fused frame to the output video
        out.write(fused_frame)

    # Release the video capture and writer
    cap.release()
    out.release()

    print(f"Output video saved to {output_video_path}")
