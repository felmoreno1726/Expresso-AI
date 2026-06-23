import argparse, os, math
import cv2
import numpy as np
from warnings import warn

import face_alignment
from skvideo.io import vread

from expresso_ai.utils.parser_helpers import str2bool
from expresso_ai.facial.landmarks import compute_video_landmarks, get_roi

FOURCC = cv2.VideoWriter_fourcc(*'MJPG')#CV_FOURCC('M','J','P','G')
FRAMES_PER_SECOND =25

def process_dataset(data_path, output_path, dimensions_out, device='cuda', compute_landmarks=False, generate_video=False, show_frame=False):
    """
    data_path: (str) path to dataset
    output_path: (str) path to output for processed dataset
    dimensions_out tuple(int): (width_out, height_out) dimensions expected to be used for image to generate landmarks from
    device: (str) device to use ('cuda', 'cpu', 'cuda:0', 'cuda:1')
    """
    #Initialize face-alignment model
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector='sfd')
    #create output paths
    os.makedirs(output_path, exist_ok=True)
    width_out, height_out = dimensions_out
    participant_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    print("Participant paths: ", participant_paths)
    for participant_path in participant_paths:
        #read participant 'original' video
        video_filename = os.path.join(participant_path, 'original.avi')
        #store landmarks in output_landmarks directory
        participant_id = int(os.path.basename(participant_path))
        print(f"Participant id: {participant_id}")
        participant_output_path = os.path.join(output_path, str(participant_id))
        os.makedirs(participant_output_path, exist_ok=True)
        #get paths to landmarks, bounding box grids and is_valid_frames
        landmarks_path = os.path.join(participant_path, 'landmark_grids.npy')
        bounding_box_path = os.path.join(participant_path, 'face_grids.npy')
        is_valid_frames_path = os.path.join(participant_path, 'is_valid_frames.npy')
        #video output path to specific directory 'width_height'
        dimensions_dir = f"{width_out}_{height_out}"
        video_output_path = os.path.join(participant_output_path, dimensions_dir)
        os.makedirs(video_output_path, exist_ok=True)
        #define output paths for videos
        face_video_path = os.path.join(video_output_path, 'face.avi')
        mouth_video_path = os.path.join(video_output_path, 'mouth.avi')
        right_eyebrow_video_path = os.path.join(video_output_path, 'right_eyebrow.avi')
        left_eyebrow_video_path = os.path.join(video_output_path, 'left_eyebrow.avi')
        right_eye_video_path = os.path.join(video_output_path, 'right_eye.avi')
        left_eye_video_path = os.path.join(video_output_path, 'left_eye.avi')
        nose_video_path = os.path.join(video_output_path, 'nose.avi')
        jaw_video_path = os.path.join(video_output_path, 'jaw.avi')
        #initialize to empty if existent
        if os.path.exists(bounding_box_path) and os.path.exists(landmarks_path) and os.path.exists(is_valid_frames_path) and not compute_landmarks:
            print("Not computing landmarks.")
            video = None
            face_grids = None
            landmark_grids = None
            is_valid_frames = None
        else:
            #read and compute
            #compute landmarks
            video = vread(video_filename)
            print("Computing landmarks")
            face_grids, landmark_grids, is_valid_frames = compute_video_landmarks(fa, video, dimensions_out)
            #face and landmark grids
            np.save(bounding_box_path, face_grids)
            np.save(landmarks_path, landmark_grids)
            np.save(is_valid_frames_path, is_valid_frames)
        #check if videos need to be generated
        if os.path.exists(face_video_path) and os.path.exists(right_eyebrow_video_path) and os.path.exists(left_eyebrow_video_path) and os.path.exists(right_eye_video_path) and os.path.exists(left_eye_video_path) and os.path.exists(nose_video_path) and os.path.exists(jaw_video_path) and not generate_video:
            print("Processed videos are found. Continuing")
            continue
        else:
            #generate video
            print("Loading face grids and landmark grids")
            video = vread(video_filename) if video is None else video
            face_grids = np.load(bounding_box_path) if face_grids is None else face_grids
            landmark_grids = np.load(landmarks_path) if landmark_grids is None else landmark_grids
            is_valid_frames = np.load(is_valid_frames_path) if is_valid_frames is None else is_valid_frames
            print("Generating video.")
            generate_participant_video(video, face_grids, landmark_grids, is_valid_frames, dimensions_out, video_output_path, show_frame)
    return


def generate_participant_video(video, face_grids, landmark_grids, is_valid_frames, dimensions_out, video_output_path, show_frame=False):
    """
    Generates face.avi, left_eye.avi, right_eye.avi videos for participant
    @param video: the 
    @param face_grids: 
    @param landmark_grids:
    @param dimensions_out:
    @param video_output_path: 
    """
    print('video output path: ', video_output_path)
    #initialize video writers
    face_video = cv2.VideoWriter(os.path.join(video_output_path, 'face.avi'), FOURCC, FRAMES_PER_SECOND, dimensions_out)
    right_eyebrow_video= cv2.VideoWriter(os.path.join(video_output_path, 'right_eyebrow.avi'), FOURCC, FRAMES_PER_SECOND, dimensions_out)
    left_eyebrow_video= cv2.VideoWriter(os.path.join(video_output_path, 'left_eyebrow.avi'), FOURCC, FRAMES_PER_SECOND, dimensions_out)
    right_eye_video= cv2.VideoWriter(os.path.join(video_output_path, 'right_eye.avi'), FOURCC, FRAMES_PER_SECOND, dimensions_out)
    left_eye_video= cv2.VideoWriter(os.path.join(video_output_path, 'left_eye.avi'), FOURCC, FRAMES_PER_SECOND, dimensions_out)
    nose_video = cv2.VideoWriter(os.path.join(video_output_path, 'nose.avi'), FOURCC, FRAMES_PER_SECOND, dimensions_out)
    jaw_video = cv2.VideoWriter(os.path.join(video_output_path, 'jaw.avi'), FOURCC, FRAMES_PER_SECOND, dimensions_out)
    #initialize frame index
    valid_frame_index = 0
    #process each frame of input video
    for i, (frame, frame_is_valid) in enumerate(zip(video, is_valid_frames)):
        if not frame_is_valid:
            print(f'Frame {i} is not valid. Skipping it in video generation')
            continue
        #process frame
        face_rect = face_grids[valid_frame_index]
        landmark_detection = landmark_grids[valid_frame_index]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #Get face Region Of Interest (ROI)
        face_roi = frame[face_rect[1]: face_rect[1] + face_rect[3], face_rect[0]: face_rect[0] + face_rect[2]]
        #display face
        #cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[0]+face_rect[2], face_rect[1]+face_rect[3]), (0, 255, 0), 2)
        #left and right eye ROI crops
        mouth_roi = get_roi(frame, landmark_detection, 'mouth', proportional_dimensions=dimensions_out, padding=0.25)
        right_eyebrow_roi = get_roi(frame, landmark_detection, 'right_eyebrow', proportional_dimensions=dimensions_out, padding=0)
        left_eyebrow_roi = get_roi(frame, landmark_detection, 'left_eyebrow', proportional_dimensions=dimensions_out, padding=0)
        right_eye_roi = get_roi(frame, landmark_detection, 'right_eye', proportional_dimensions=dimensions_out, padding=0.25)
        left_eye_roi =get_roi(frame, landmark_detection, 'left_eye', proportional_dimensions=dimensions_out, padding=0.25)
        nose_roi = get_roi(frame, landmark_detection, 'nose', proportional_dimensions=dimensions_out, padding=0)
        jaw_roi = get_roi(frame, landmark_detection, 'jaw', proportional_dimensions=dimensions_out, padding=0)
        if show_frame:
            cv2.imshow("frame", frame)
            cv2.imshow("Mouth", mouth_roi)
            cv2.imshow("Left eyebrow", left_eyebrow_roi)
            cv2.imshow("Right eyebrow", right_eyebrow_roi)
            cv2.imshow("Left eye", left_eye_roi)
            cv2.imshow("Right eye", right_eye_roi)
            cv2.imshow("Nose", nose_roi)
            cv2.imshow("Jaw", jaw_roi)
        #resize frame to desired output
        face_roi  = cv2.resize(face_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)
        mouth_roi  = cv2.resize(mouth_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)
        right_eyebrow_roi = cv2.resize(right_eyebrow_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)
        left_eyebrow_roi = cv2.resize(left_eyebrow_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)
        right_eye_roi = cv2.resize(right_eye_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)#INTER_CUBIC)#INTER_LANCZOS4)
        left_eye_roi = cv2.resize(left_eye_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)#INTER_CUBIC)#INTER_LANCZOS4)
        nose_roi = cv2.resize(nose_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)
        jaw_roi = cv2.resize(jaw_roi, dimensions_out, interpolation=cv2.INTER_LANCZOS4)
        #write out videos
        face_video.write(face_roi)
        right_eyebrow_video.write(right_eyebrow_roi)
        left_eyebrow_video.write(left_eyebrow_roi)
        right_eye_video.write(right_eye_roi)
        left_eye_video.write(left_eye_roi)
        nose_video.write(nose_roi)
        jaw_video.write(jaw_roi)
        if show_frame:
            cv2.waitKey(1)
        #update frame count
        valid_frame_index += 1
    #release video resources
    face_video.release()
    right_eyebrow_video.release()
    left_eyebrow_video.release()
    right_eye_video.release()
    left_eye_video.release()
    nose_video.release()
    jaw_video.release()
    #landmark_grids_length = len(landmark_grids)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses command line inputs to face and landmark pre-processing routine.")
    parser.add_argument('--data_path', nargs='?', default='./full_dataset', type=str, help="Default='./full_dataset'. Path to processed dataset")
    parser.add_argument('--output_width', nargs='?', default=112, type=int, help="Default=112. Output width dimension of the desired landmarks")
    parser.add_argument('--output_height', nargs='?', default=112, type=int, help="Default=112. Output hegiht dimension of the desired landmarks")
    parser.add_argument('--device', nargs='?', default='cuda', type=str, help="Default='cuda'. Defines the device to use for computing face detection and landmark extraction. Select from: cpu, cuda, cuda:0, cuda:1, ...")
    #routine parameters
    parser.add_argument('--compute_landmarks', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Flag deterines if processing is done using pre-computed landmarks, or freshly computed landmarks. If landmarks for video have already been computed and flag is false, then landmarks are loaded, otherwise landmarks are computed.")
    parser.add_argument('--generate_video', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Flag deterines if video generation is skipped for participants in which video file already exists. If video has already been computed and flag is false, then landmarks are loaded, otherwise landmarks are computed.")
    parser.add_argument('--show_frame', type=str2bool, nargs='?', const=True, default=False, help="Default=False. Flag determines if frames are displayed while video generation computation is ongoing")
    args = parser.parse_args()
    process_dataset(data_path=args.data_path, output_path=args.data_path, dimensions_out=(args.output_width, args.output_height), device=args.device, compute_landmarks=args.compute_landmarks, generate_video=args.generate_video, show_frame=args.show_frame)

