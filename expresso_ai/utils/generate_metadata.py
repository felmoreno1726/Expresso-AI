import argparse, json, os, random
import cv2
import pandas as pd

from .video_helpers import data_samples_from_num_frames


DISTRIBUTION_RANGES = [(0, 13), (14, 19), (20, 28), (29, 63)]


def find_range(score):
    """
    Find the range in which score falls in.
    @param score the value assigned to a particular label.
    @return the index of the range in DISTRIBUTION_RANGES in which the score falls into.
    """
    for idx, range in enumerate(DISTRIBUTION_RANGES):
        if range[0] <= score <= range[1]:
            return idx
    raise Exception("No range was found for score {}".format(score))


def get_num_samples(data_path):
    labels_df = pd.read_csv(os.path.join(data_path, 'labels.csv'))
    return len(labels_df)


def get_participant_samples(data_path, window_size=300, overlap=0.5):
    labels_df = pd.read_csv(os.path.join(data_path, 'labels.csv'))
    sample_indices = [str(i) for i in range(get_num_samples(data_path))]
    participant_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,f)) and f in sample_indices]
    #initialize samples column
    labels_df['samples'] = None
    for participant_path in participant_paths:
        participant_id = os.path.basename(participant_path)
        participant_frames = get_frames(participant_path, dimensions_dir)
        participant_samples = data_samples_from_num_frames(participant_frames, window_size, overlap)
        print('(id, samples): ({},{})'.format(participant_id, participant_samples))
        labels_df.at[int(participant_id), 'samples'] = participant_samples
    print('labels df: \n', labels_df)
    labels_df.to_csv('./paticipant_with_samples.csv')
    return labels_df
    
    
def generate_phase_metadata(data_path, img_size, sample_indices, window_size = 30, overlap = 0.5, input_dilation=1):
    w, h = img_size
    dimensions_dir = f"{w}_{h}"
    participant_paths = [os.path.join(data_path, f) for f in sample_indices if os.path.isdir(os.path.join(data_path,f))]
    labels_path = os.path.join(data_path, 'labels.csv')
    labels_df = pd.read_csv(labels_path)
    labels = []
    sample_to_id = []
    sample_to_window = []
    id_to_samples = {}
    total_samples = 0
    #positive_samples = 0
    participant_distribution = [0, 0, 0, 0] 
    sample_distribution = [0, 0, 0, 0] 
    for participant_path in participant_paths:
        #get the number of data samples of current video
        participant_frames = get_frames(participant_path, dimensions_dir)
        participant_samples = data_samples_from_num_frames(participant_frames, window_size, overlap)
        participant_id = int(os.path.basename(participant_path))
        participant_label = get_label(participant_id, labels_df)
        #add indices to metadata
        participant_label = get_label(participant_id, labels_df)
        print('(id, samples, frames, label): ({}, {}, {}, {})'.format(participant_id, participant_samples, participant_frames, participant_label))
        #add pariticpant to metadata
        labels.extend([participant_label] * participant_samples)
        sample_to_id.extend([participant_id] * participant_samples)
        sample_to_window.extend(list(range(participant_samples)))
        id_to_samples[participant_id] = [total_samples + sample_index for sample_index in range(participant_samples)]
        #increase total samples
        total_samples += participant_samples
        participant_distribution[find_range(participant_label)] += 1
        sample_distribution[find_range(participant_label)] += participant_samples
    print('participant distribution: ', participant_distribution)
    print('sample distribution: ', sample_distribution)
    print('phase samples: ', total_samples)
    total_samples_in_ids = samples_in_ids(id_to_samples)
    #assert data is not-corrupted
    assert total_samples == len(labels) == len(sample_to_id) == len(sample_to_window) == total_samples_in_ids, "Mismatch in samples size. total samples {}, total labels {}, samples to ids {}, samples to window {}, samples to ids {}".format(total_samples, len(labels), len(sample_to_id), len(sample_to_window), total_samples_in_ids)
    metadata = {
            "num_participants": len(participant_paths),
            "samples":          total_samples,
            "window_size":      window_size,
            "overlap":          overlap,
            "input_dilation":   input_dilation,
            "labels":           labels,
            "sample_to_id":     sample_to_id,
            "sample_to_window": sample_to_window,
            "id_to_samples":    id_to_samples
    }
    return metadata, sample_distribution

def balance_metadata(metadata, samples_distribution):
    if 0 in samples_distribution:
        raise UnbalanceableDataError
    total_samples = sum(samples_distribution)
    print("total samples pre-upsampling: ", total_samples)
    upsampling_target = max(samples_distribution)
    total_balanced_samples = len(DISTRIBUTION_RANGES) * upsampling_target
    print("upsampling target: ", upsampling_target)
    print("total balanced_samples: ", total_balanced_samples)
    #total_balanced_samples = 4 * upsampling_target
    classes_to_sample = [upsampling_target - samples_distribution_class for samples_distribution_class in samples_distribution]
    print('classes to sample: ', classes_to_sample)
    if sum(classes_to_sample) == 0:#no classes need re-sampling
        return metadata
    #else we balance using sampling
    #get random samples to balance the metadata
    balancing_labels, balancing_sample_to_id, balancing_sample_to_window, balancing_id_to_samples = \
            balance_sampling(classes_to_sample, metadata)
    #generate balanced metadata with random samples
    balanced_labels = metadata["labels"] + balancing_labels
    balanced_sample_to_id = metadata["sample_to_id"] + balancing_sample_to_id
    balanced_sample_to_window = metadata["sample_to_window"] + balancing_sample_to_window
    balanced_id_to_samples = append_dictionary_lists(metadata["id_to_samples"], balancing_id_to_samples)
    assert total_balanced_samples == len(balanced_labels) == len(balanced_sample_to_id) == len(balanced_sample_to_window) == samples_in_ids(balanced_id_to_samples), "The total samples in the metadata should match. But got ({}, {}, {}, {}, {})".format(total_balanced_samples, len(balanced_labels), len(balanced_sample_to_id), len(balanced_sample_to_window), samples_in_ids(balanced_id_to_samples))
    balanced_metadata = {
            "num_participants": metadata["num_participants"],
            "samples": total_balanced_samples,
            "window_size": metadata["window_size"],
            "overlap": metadata["overlap"],
            "input_dilation": metadata["input_dilation"],
            "labels" : balanced_labels,
            "sample_to_id": balanced_sample_to_id,
            "sample_to_window": balanced_sample_to_window,
            "id_to_samples": balanced_id_to_samples
    }
    return balanced_metadata

def balance_sampling(classes_to_sample, metadata):
    """
    Samples from the metatada to obtain balanced range distributions
    @param classes_to_sample is an int list of length DISTRIBUTION_RANGES such that classes_to_sample[i] is the number of samples to generate for range class DISTRIBUTION_OF_RANGES[i]. Requires all entries of classes_to_sample be non-negative
    """
    assert [entry >=0 for entry in classes_to_sample]
    balancing_labels = []
    balancing_sample_to_id = []
    balancing_sample_to_window = []
    balancing_id_to_samples = {}
    current_sample = metadata["samples"]
    #Seed the random sample generation for reproducibility purposes
    random.seed(1)
    while sum(classes_to_sample) != 0:
        sample = random.randint(0, metadata["samples"] - 1)
        score = metadata["labels"][sample]
        range_class = find_range(score)
        if classes_to_sample[range_class] != 0:
            #add to the points sampled
            participant_id = metadata["sample_to_id"][sample]
            balancing_labels.append(score)
            balancing_sample_to_id.append(participant_id)
            balancing_sample_to_window.append(metadata["sample_to_window"][sample])
            balancing_id_to_samples[participant_id] = balancing_id_to_samples.get(participant_id, []) + [current_sample]
            #update the classes_to_sample to reflect changes in already sampled points
            classes_to_sample[range_class] -= 1
        else:
            #sampled point class does not require balancing
            continue
    return balancing_labels, balancing_sample_to_id, balancing_sample_to_window, balancing_id_to_samples

def append_dictionary_lists(a, b):
    """
    a: dictionary object with lists as values
    b: a dictionary of lists as values where each key in b is a key in a
    Mutates a to append every list in b to a the corresponding list in a.
    Returns the mutated a
    """
    for key, val_list in b.items():
        a[key] = a[key] + val_list
    return a

def get_label(id_number, df):
    return int(df[df['id'] == id_number]['label'])

def samples_in_ids(id_to_samples):
    total = 0
    for samples in id_to_samples.values():
        total += len(samples)
    return total

def get_frames(participant_path, dimensions_dir):
    """
    Gets the number of frames of the video data in the participant_path. 
    Asserts that data is valid and all videos of the participant have the same number of frames
    @param participant_path (str): path to participant directory. 
    @param dimensions_dir (str): name of directory that contains video within participant_path. Must contain 'face.avi', 'left_eye.avi', 'right_eye.avi' and lengths of all 3 videos must match.
    @return (int) the number of frames in the participant videos.
    """
    #read videos
    face_video =     cv2.VideoCapture(os.path.join(participant_path, dimensions_dir, "face.avi"))
    left_eye_video = cv2.VideoCapture(os.path.join(participant_path, dimensions_dir, "left_eye.avi"))
    right_eye_video= cv2.VideoCapture(os.path.join(participant_path, dimensions_dir, "right_eye.avi"))
    #check that the videos have the same number of frames
    face_frames = face_video.get(cv2.CAP_PROP_FRAME_COUNT)
    left_eye_frames = left_eye_video.get(cv2.CAP_PROP_FRAME_COUNT)
    right_eye_frames = right_eye_video.get(cv2.CAP_PROP_FRAME_COUNT)
    assert face_frames == left_eye_frames == right_eye_frames, "Number of video frames don't match. Face frames {}, left eye frames {}, right eye frames {}".format(face_frames, left_eye_frames, right_eye_frames) 
    return int(face_frames)


def save_metadata(metadata, output_path):
    with open(output_path, 'w') as f:
        json.dump(metadata, f)
    print("metadata saved to {}".format(output_path))

def generate_metadata(window_size, overlap, input_dilation, data_path, img_size, train_indices, val_indices, test_indices):
    #generate and balance metadata
    #train
    print("\n Generating training metadata")
    train_metadata, train_sample_distribution = generate_phase_metadata(data_path, img_size, train_indices, window_size=window_size, overlap=overlap, input_dilation=input_dilation)
    #train_metadata = balance_metadata(train_metadata, train_sample_distribution)
    #validation
    print("\n Generating validation metadata")
    val_metadata, val_sample_distribution = generate_phase_metadata(data_path, img_size, val_indices, window_size = window_size, overlap=overlap, input_dilation=input_dilation)
    #val_metadata = balance_metadata(val_metadata, positive_val_samples, negative_val_samples)
    #test
    print("\n Generating testing metadata")
    test_metadata, test_sample_distribution = generate_phase_metadata(data_path, img_size, test_indices, window_size = window_size, overlap=overlap, input_dilation=input_dilation)
    #test_metadata = balance_metadata(test_metadata, positive_test_samples, negative_test_samples)
    #interpret
    print("\n Generating interpretation metadata")
    #uses all participants without any balancing
    interpret_indices = test_indices
    #interpret_indices = [str(i) for i in range(get_num_samples(data_path))]
    interpret_metadata = generate_phase_metadata(data_path, img_size, interpret_indices, window_size=window_size, overlap=overlap, input_dilation=input_dilation)[0]
    #combine generated metadata into one file
    metadata = {
            'train': train_metadata,
            'val' : val_metadata,
            'test': test_metadata,
            'interpret': interpret_metadata
    }
    return metadata

def get_training_indices(data_path, val_indices, test_indices):
    train_indices = set([str(i) for i in range(get_num_samples(data_path))])####TODO: generalize. train indices need not be ints [0, num_samples]
    train_indices.difference_update(set(val_indices))
    train_indices.difference_update(set(test_indices))
    return train_indices

class UnbalanceableDataError(Exception):
    """Raised when the dataset provided cannot be balanced"""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parses paths to data')
    parser.add_argument('--data_path', nargs='?', default='./full_dataset', type=str, help="Path to processed dataset. ")
    parser.add_argument('--window_size', type=int, default=300, help="The number of frames in each data sample.")
    parser.add_argument('--overlap', type=float, default=0.5, help="The percent of overlap between continuous data points in a video sample.")
    parser.add_argument('--val_indices', nargs='+', default=[])
    parser.add_argument('--test_indices', nargs='+', default=[])
    #get cmd line arguments
    args = parser.parse_args()
    print('args: ', args)
    print("data_path: ", args.data_path)
    print("overlap: ", args.overlap)
    print("window_size: ", args.window_size)
    print("validation indices: ", args.val_indices)
    print("test indices: ", args.test_indices)
    #get train samples
    train_indices = get_training_indices(args.data_path, args.val_indices, args.test_indices)
    print("train indices: ", train_indices)
    #generate train/val/test metadata
    metadata = generate_metadata(args.window_size, args.overlap, args.data_path, train_indices, args.val_indices, args.test_indices)
    save_metadata(metadata, os.path.join(args.data_path, 'metadata.json'))
