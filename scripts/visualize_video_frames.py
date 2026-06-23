import os
import cv2
import argparse

def show_participant(path_to_sample_dir, visualization_type):
    #display attributions graph
    graph_image_path = os.path.join(path_to_sample_dir, 'per_channel_attributions.png')
    print("graph image path: ", graph_image_path)
    graph_image = cv2.imread(graph_image_path)
    cv2.imshow('per_channel_attributions', graph_image)
    filename = os.path.join(path_to_sample_dir, visualization_type)
    print("filename: ", filename)
    cap = cv2.VideoCapture(filename)
    #cap.set(cv2.CV_CAP_PROP_FOURCC, cv2.CV_FOURCC('H', '2', '6', '5'))
    frame_num = 0
    #Read until video is completed
    while(cap.isOpened()):
        #Display frame number
        frame_num += 1
        print("Frame: ", frame_num)
        cv2.imshow("Frame: ", frame_num)
        #read and show video frame
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("Frame: ", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            wait_for_input = input("Press enter to continue")
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes in as input the participant to display")
    parser.add_argument('--sample_dir', type=str, help="path to participant's sample director")
    parser.add_argument('--visualization_type', type=str, help="The type of visualiztion to provide")
    args= parser.parse_args()
    show_participant(args.sample_dir, args.visualization_type)

