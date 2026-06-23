import argparse, os, sys, subprocess


def extract_action_units(data_path, output_path, binaries_path):
	#add binaries path to working directory path
	sys.path.append(binaries_path)
	feature_extraction_path =os.path.join(binaries_path, "FeatureExtraction")
	#create output_dir if it does not exist
	os.makedirs(output_path, exist_ok=True)
	#iterate over participants to extract action units from
	participant_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
	for participant_path in participant_paths:
		participant_id = str(os.path.basename(participant_path))
		participant_video_path = os.path.join(participant_path, 'original.avi')
		#set up output path for participant
		participant_out_path = os.path.join(output_path, participant_id)
		#extract action units from it
		cmd = "{feature_extraction_path} -f {participant_video_path}  -out_dir {participant_out_path} -of aus.csv -aus".format(
			feature_extraction_path=feature_extraction_path,
			participant_video_path=participant_video_path,
			participant_out_path=participant_out_path
		)
		subprocess.call(cmd, shell=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parses command line inputs to action unit generation program.")
	parser.add_argument('--data_path', nargs='?', default='./full_dataset', type=str, help="Default='./full_dataset'. Path to processed dataset")
	parser.add_argument('--output_path', nargs='?', default='AUs', type=str, help="Default='./AUs'. Path to extract action units to")
	parser.add_argument('--OpenFace_binaries_path', nargs='?', default='/home/openface-build/build/bin', type=str, help="Default='/home/openface-build/build/bin. Path to openface executable binaries.")
	args = parser.parse_args()
	extract_action_units(args.data_path, args.output_path, binaries_path=args.OpenFace_binaries_path)
