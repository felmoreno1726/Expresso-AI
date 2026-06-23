import os, shutil, argparse
import pandas as pd
import numpy as np

from scipy.io import loadmat


def process_all_videos(input_dir, output_dir):
    labels_filepath = os.path.join(output_dir, 'labels.csv')
    df = pd.read_csv(labels_filepath)
    for row_num, row in df.iterrows():
        filename = row["file_path"]
        file_id = row["id"]
        file_path = os.path.join(input_dir, filename)
        sample_dir = os.path.join(output_dir, str(file_id))
        os.makedirs(sample_dir, exist_ok=True)
        output_sample = os.path.join(sample_dir, 'original.avi')
        print(f"Copying {filename} to {output_sample}")
        shutil.copy(file_path, output_sample)

if __name__ == "__main__":
    #path_to_metadata = "../reference_metadata.mat"
    #mean = get_average_face_grid(path_to_metadata)
    #print(mean)
    parser = argparse.ArgumentParser(description="parses the prepraration input/output paths")
    parser.add_argument('--input_path', type=str, help="Path to the raw data directory")
    parser.add_argument('--output_path',type=str, help="Path to the output data directory")
    args = parser.parse_args()
    #get input/output paths
    process_all_videos(args.input_path, args.output_path)
