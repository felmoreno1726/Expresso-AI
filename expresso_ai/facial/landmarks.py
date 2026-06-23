import os, collections
from warnings import warn
from collections import OrderedDict

#import imutils
from imutils import face_utils
import numpy as np
import cv2
import torch
import face_alignment
import matplotlib.pyplot as plt
from skvideo.io import vread

from ..utils.roi_helpers import preserve_proportions, pad_rectangle_box

LANDMARK_DIMENSIONS = (68,2)
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


def compute_video_landmarks(fa, video, dimensions_out):
    """
    For each frame in video, detects the face with highest confidence, extracts the frame with 
    """
    #initialize face and landmark grids
    face_grids = None
    landmark_grids = None
    is_valid_frames = []
    for frame in video:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #extract face bounding box
        detected_faces = fa.face_detector.detect_from_image(frame)
        if len(detected_faces) < 1:
            #there were no faces detected
            warn("Detected faces is empty. Skipping current frame")
            is_valid_frames.append(0)
            continue
        #use face with highest confidence
        face_detection = detected_faces[0]
        confidence_score = face_detection[-1]
        if confidence_score < 0.75:
            warn(f"Face detected has a low confidence score: ({confidence_score}). Skipping current frame")
            is_valid_frames.append(0)
            continue
        #express face bounding box as two points
        face_bbox = np.array([[face_detection[0], face_detection[1]], [face_detection[2], face_detection[3]]])
        #pad face
        face_rect = cv2.boundingRect((face_bbox))
        face_rect = preserve_proportions(face_rect, dimensions_out)
        #face_rect = pad_rectangle_box(face_rect, 0.1, mode='length')
        #face_roi = frame[face_rect[1]: face_rect[1] + face_rect[3], face_rect[0]: face_rect[0] + face_rect[2]]
        #display face
        #cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[0]+face_rect[2], face_rect[1]+face_rect[3]), (0, 255, 0), 2)
        #get landmarks
        landmark_detections = fa.get_landmarks_from_image(frame, [face_detection])
        landmark_detection = landmark_detections[0]
        #write out to grids
        face_grids = np.concatenate((face_grids, np.array([face_rect])), axis=0) if face_grids is not None else np.array([face_rect])
        landmark_grids = np.concatenate((landmark_grids, [landmark_detection]), axis=0) if landmark_grids is not None else np.array([landmark_detection])
        is_valid_frames.append(1)
    assert len(video) == len(is_valid_frames) >= len(face_grids) == len(landmark_grids), "Lengths of video, is_valid_frames, face_grids and landmark_grids are not as expected."
    return face_grids, landmark_grids, is_valid_frames


def get_roi(frame, landmarks, region_name, proportional_dimensions=None, padding=0, padding_mode='length'):
    """
    Gets the region of interest of a given frame. Given the landmarks,
    the function extracts the bounding box of the region such that 
    dimmensions are proportional and applies padding afterwards.
    @params
    region_name: (str) the name of the region to extract from the landmarks
    proportional_dimensions: None or tuple(int, int) dimensions that are meant to be preserved by the resulting bounding box.
    padding: (int, float) the value of padding to apply to the already proportional bounding box
    padding_mode: (str) from 'pixel', 'length', 'area'
    """
    i, j = face_utils.FACIAL_LANDMARKS_IDXS[region_name]
    bbox = landmarks[i: j]
    rect = cv2.boundingRect(bbox)
    (x, y, w, h) = rect
    #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)###NOTE: Uncomment for debugging
    if proportional_dimensions is not None:
        #get a tight region with proportions
        rect = preserve_proportions(rect, proportional_dimensions)
        (x, y, w, h) = rect
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,112,112), 2)###NOTE: Uncomment for debugging
    rect = pad_rectangle_box(rect, padding=padding, mode=padding_mode)
    (x, y, w, h) = rect
    roi = frame[y: y+h, x: x+w]
    #cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)###NOTE: Uncomment for debugging
    if np.prod(roi.shape) == 0:
        warn(f"ROI {region_name} has area 0. Replacing with a black frame.")
        roi = np.zeros((w, h, 3), dtype=np.uint8)
    return roi


def get_video_landmarks_with_batches(fa, video, batch_size=1):
    warn("get_video_landmarks_with_bacthes is deprecated. Use compute_video_landmarks method")
    video = np.transpose(video, (0,-1,1,2))#T,W,H,C -> T,C,W,H
    video = torch.from_numpy(video).float().to(device)
    num_batches = video_length // batch_size
    video_landmarks = None
    for batch_id in range(num_batches):
        batch_index = batch_id * batch_size
        this_batch_size = min(batch_size, video_length - batch_index)
        batch = video[batch_index: batch_index + batch_size]
        try:
            landmark_detections = fa.get_landmarks_from_batch(batch)
        except ValueError as e:
            warn("Error {} trying to detect landmarks from batch".format(e))
            landmark_detections = np.zeros((this_batch_size, LANDMARK_DIMENSIONS[0], LANDMARK_DIMENSIONS[1]))
        import pdb; pdb.set_trace()
        video_landmarks = np.concatenate((video_landmarks, np.array(landmark_detections))) if video_landmarks is not None else np.array(landmark_detections)
    return video_landmarks


def get_video_landmarks(fa, video):
    warn("get_video_landmarks is deprecated. Use compute_video_landmarks method")
    print("video shape: ", video.shape)
    video_length, w, h, channels = video.shape
    video_landmarks = None
    for frame in video:
        detections = fa.get_landmarks_from_image(frame)
        if detections is None:
            detection = np.zeros(LANDMARK_DIMENSIONS)
        else:
            if len(detections) > 1:
                warn("Unexpected number of detections {}".format(len(detections)))
            detection = detections[0]
            #import pdb; pdb.set_trace()
            video_landmarks = np.concatenate((video_landmarks, np.array([detection]))) if video_landmarks is not None else np.array([detection])
    return video_landmarks


