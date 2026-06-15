"""
Video I/O utilities for reading and writing football video analysis.

This module provides simple wrapper functions for OpenCV video operations
to standardize reading input videos and writing processed output videos.

Key functions:
- read_video: Load all frames from a video file with FPS information
- save_video: Write analyzed frames back to video file with correct encoding
"""

import cv2
import numpy as np


def read_video(video_path):
    """
    Read all frames from a video file.
    
    Args:
        video_path (str): Path to input video file
        
    Returns:
        tuple: (frames list, fps) where frames are numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Read frames until end of video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Get frames-per-second for output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return frames, fps


def save_video(frames, output_path, fps):
    """
    Write processed frames to an output video file.
    
    Args:
        frames (list): List of frame numpy arrays
        output_path (str): Path for output video file
        fps (float): Frames per second for output video
    """
    if len(frames) == 0:
        return

    # Extract frame dimensions from first frame
    height, width = frames[0].shape[:2]

    # Use mp4v codec for H.264 MP4 output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # Write all frames to output video
    for frame in frames:
        out.write(frame)

    out.release()