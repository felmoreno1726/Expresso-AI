import os, argparse
import numpy as np

import face_alignment
from skvideo.io import vread

def generate_dataset_landmarks(data_path, output_path, dimensions_out = (0,0), device='cuda'):
    """
    dimensions_out tuple(int): (width_out, height_out) dimensions expected to be used for image to generate landmarks from
    """
    warn("get_video_landmarks_with_bacthes is deprecated. Use compute_video_landmarks method")
    #create output paths
    os.makedirs(output_path, exist_ok=True)
    width_out, height_out = dimensions_out
    if dimensions_out == (0,0):
        dimensions_dir = "original"
    else:
        dimensions_dir = f"{width_out}_{height_out}"
    landmarks_output_path = os.path.join(output_path, dimensions_dir)
    os.makedirs(landmarks_output_path, exist_ok=True)
    #generate landmarks for the dataset
    participant_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    fa = load_landmarks_model(device)
    for participant_path in participant_paths:
        print("Generating landmarks for participant: ", participant_path)
        #read participant video
        video_filename = os.path.join(participant_path, 'face.avi')
        video = vread(video_filename)
        video_length, w, h, channels = video.shape
        video_landmarks = get_video_landmarks(fa, video)
        #check that video and landmarks match in duration
        landmarks_length = len(video_landmarks)
        assert landmarks_length == video_length, f"Number of landmarks {landmarks_length} and number of frames {video_duration} in video don't match. They must be aligned."
        #store landmarks in output_landmarks directory
        participant_id = int(os.path.basename(participant_path))
        participant_output_path = os.path.join(landmarks_output_path, str(participant_id))
        os.makedirs(participant_output_path, exist_ok=True)
        np.save(os.path.join(participant_output_path, 'landmarks.npy'), video_landmarks)
    return

def load_landmarks_model(device):
    """
    device (str): 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
    """
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device=device,
        face_detector='sfd')
    return fa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses command line inputs to landmark extraction program")
    parser.add_argument('--data_path', nargs='?', default='./full_dataset', type=str, help="Path to processed dataset")
    parser.add_argument('--output_path',nargs='?', default='./landmarks', type=str, help="Path to store extracted landmarks to.")
    parser.add_argument('--output_width', nargs='?', default=0, type=int, help="Output width dimension of the desired landmarks")
    parser.add_argument('--output_height', nargs='?', default=0, type=int, help="Output hegiht dimension of the desired landmarks")
    args = parser.parse_args()
    generate_dataset_landmarks(args.data_path, args.output_path, dimensions_out=(args.output_width, args.output_height))
