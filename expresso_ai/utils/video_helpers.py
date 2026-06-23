import os
import torch
import numpy as np
from functools import lru_cache


def data_samples_from_num_frames(num_frames, window_size, overlap):
    """
    Function computes the number of samples found in a video given:
    @param num_frames (int): the number of frames in a video given
    @param window_size (int): the number of frames that a sample spans
    @param overlap (float): the percentage of overlap between samples gathered from videos.
    Derivation: 
        (1 - overlap) * window_size * samples + overlap * window_size <= num_frames
    Therefore:
        samples <= (num_frames - overlap * window_size) / ((1 - overlap) * window_size)
    @returns (int) positive number of samples found    
    """
    return max(0, int((num_frames - overlap * window_size) // ((1 - overlap) * window_size)))

def get_initial_frame(window_size, window_index, overlap):
    return int((1 - overlap) * window_size * window_index)

def get_frame_indices(window_size, window_index, overlap, input_dilation):
    initial_frame = get_initial_frame(window_size, window_index, overlap)
    print(initial_frame, window_index, input_dilation) 
    return np.arange(initial_frame, initial_frame + window_size, input_dilation)

def read_frames(videos, initial_frame, window_size, input_dilation):
    """
    Returns the image of the current frame in RGB format, passed by the pytorch transorm object
    """
    video_frames = {video_name: video_frames[initial_frame: initial_frame + window_size: input_dilation] for video_name, video_frames in videos.items()}
    return video_frames

def frames_per_sample(window_size, input_dilation):
    return window_size // input_dilation

def transform_frames(multi_video_frames, transforms):
    """
    multi_video_frames: (Dict) of keys {'face', 'left_eye', 'right_eye'} and values corresponding the to video frames.
    transforms: (Dict) of keys  {'face', 'left_eye', 'right_eye'} and values corresponding to the pytorch frame transformers.
    returns the result of applying the corresponding transform along the images in the frame sequence
    """
    transformed_video_frames = {
            video_name: np.stack([
                transforms[video_name](video_frame) for video_frame in video_frames
            ])
            for video_name, video_frames in multi_video_frames.items()
    }
    return transformed_video_frames

def transform_videos(multi_videos, transforms):
    """
    multi_video_frames: (Dict) of keys video_names as keys and values corresponding videos
    transforms: (Dict) of keys  {'face', 'left_eye', 'right_eye'} and values corresponding to the pytorch frame transformers.
    returns the result of applying the corresponding transform to the video
    """
    transformed_videos= {
            video_name: 
                transforms[video_name](video)
                   for video_name, video in multi_videos.items()
    }
    return transformed_videos

class ResizeVideo(object):
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, vid):
        return resize(vid, self.size, interpolation=self.interpolation)

# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='nearest'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)
