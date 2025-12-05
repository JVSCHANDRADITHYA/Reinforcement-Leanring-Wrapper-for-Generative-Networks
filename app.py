import os
import torch
import numpy as np
import streamlit as st
from options.test_options import TestOptions
from models import create_model
from util import util
import cv2
from datetime import datetime
from PIL import Image
import io

# Define hard-coded parameters and paths
results_dir = './results/'            # Directory to save results
model_name = 'derain'  # Model name
model_type = 'test'                   # Model type
no_dropout = True                     # No dropout flag

def process_image(image):
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

    # Convert the image to OpenCV format
    input_image = np.array(image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Preprocess the image
    original_size = input_image.shape[:2]  # Save original size (height, width)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image_resized = cv2.resize(input_image_rgb, (256, 256))
    input_image_resized = np.asarray([input_image_resized])
    input_image_resized = np.transpose(input_image_resized, (0, 3, 1, 2))
    data = {"A": torch.FloatTensor(input_image_resized)}

    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference

    result_image = model.get_current_visuals()['fake']
    result_image = util.tensor2im(result_image)
    result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

    # Resize result image back to original resolution
    result_image = cv2.resize(result_image, (original_size[1], original_size[0]))

    # Resize the input image to match the result image dimensions for proper fusion
    input_image_resized = cv2.resize(input_image, (result_image.shape[1], result_image.shape[0]))

    # Fuse the input image with the result image side-by-side
    fused_image = np.hstack((input_image_resized, result_image))

    return result_image, fused_image

def save_image(image, filename):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img.save(filename)

def main():
    st.title("Image Processing with Model")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process Image'):
            result_image, fused_image = process_image(image)

            # Generate unique filenames using timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_path = os.path.join(results_dir, f'result_image_{timestamp}.png')
            fused_image_path = os.path.join(results_dir, f'input_output_fused_{timestamp}.png')

            # Save the result and fused images
            save_image(result_image, result_path)
            save_image(fused_image, fused_image_path)

            st.image(result_image, caption='Result Image', use_column_width=True)
            st.image(fused_image, caption='Fused Image', use_column_width=True)

            st.write(f"Result saved to {result_path}")
            st.write(f"Fused image saved to {fused_image_path}")

if __name__ == "__main__":
    main()
