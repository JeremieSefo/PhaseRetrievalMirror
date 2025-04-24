import cv2
import numpy as np
from IPython.display import display, Image, Video
from PIL import Image as PILImage
import io

def create_video(image_sequence_real, image_sequence_imag, output_file, fps=10):
    assert len(image_sequence_real) == len(image_sequence_imag), "Real and imaginary sequences must have the same length"

    height, width = image_sequence_real[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width*2, height)) # width*2 for real and imaginary parts side by side

    for real_part, imag_part in zip(image_sequence_real, image_sequence_imag):
        norm_imag_part = imag_part  # Normalize if needed (you can add normalization here)
        combined_image = np.hstack((real_part, norm_imag_part))
        combined_image = cv2.cvtColor(combined_image.astype('uint8'), cv2.COLOR_GRAY2BGR)
        out.write(combined_image)

    out.release()

def display_video(video_file, fps=10):
    """Displays video inline in Jupyter Notebook using IPython's display function."""
    # Use IPython's Video display
    display(Video(video_file, embed=True, width=640, height=480))

def update_frame_from_video(video_file, frame_idx, zoom_factor=2):
    """Extracts a frame from the video and displays it."""
    cap = cv2.VideoCapture(video_file)
    
    # Set the frame to the specified index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = cv2.resize(frame, (frame.shape[1] * zoom_factor, frame.shape[0] * zoom_factor))  # Apply zoom
        
        # Convert the frame into an image format that can be displayed in Jupyter
        pil_image = PILImage.fromarray(frame)
        byte_array = io.BytesIO()
        pil_image.save(byte_array, format='PNG')
        byte_array.seek(0)

        display(Image(data=byte_array.read(), format='png'))

    cap.release()

from moviepy import VideoFileClip

def convert_to_mp4(input_file, output_file):
    clip = VideoFileClip(input_file)
    clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
