import os, multiprocessing
import pandas as pd
import numpy as np
from warnings import warn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
#Define 'agg' backend of matplotlib for usage over ssh
#matplotlib.use('AGG') ###UNCOMMENT THIS LINE IF YOU DON'T WANT THE IMAGES GENERATED TO BE SHOWN
pd.options.display.float_format = '{:,.2f}'.format

import torch
import captum

#from captum.attr import visualization as viz
from ..facial.landmarks import get_roi, FACIAL_LANDMARKS_IDXS
from ..facial.action_units import load_participant_AU, get_sample_AUS, get_effective_AUS, compute_au_correlations
from ..utils import visualization as viz
from ..utils.video_helpers import get_frame_indices
from ..utils.AverageMeter import AverageMeter
from ..utils.ForkedPdb import ForkedPdb


class ModelInterpreter():

    FRAMES_PER_SECOND = 25
    N = 5
    CLUSTERS = {
            "minimal": (0, 13),
            "mild": (14, 19),
            "moderate": (20, 28),
            "severe": (29, 63)
    }

    def __init__(self, model, compute_attributions=True, algorithm_name='DeepLift', visualization_methods=['heap_map'], signs=['absolute_value']):
        if model.training:
            warn("Expected model to be in evaluation mode. Setting mode to evaluation")
            model.eval()
        self.has_convergence_delta = True
        self.model = model
        self.compute_attributions = compute_attributions
        self.algorithm_name = algorithm_name
        self.visualization_methods = visualization_methods
        self.signs = signs
        #initialize the attribution methods
        if algorithm_name == 'Saliency':
            from captum.attr import Saliency
            self.interpretation_algorithm = Saliency(model)
        elif algorithm_name == 'DeepLift': 
            from captum.attr import DeepLift
            self.interpretation_algorithm = DeepLift(model)
        elif algorithm_name == 'IntegratedGradients':
            from captum.attr import IntegratedGradients
            self.interpretation_algorithm = IntegratedGradients(model)
        elif algorithm_name == 'NoiseTunnel':
            from captum.attr import NoiseTunnel
            self.interpretation_algorithm = NoiseTunnel(integrated_gradients)
        else:
            raise NameError(f"{algorithm_name} not defined")
    
    @classmethod
    def get_input_names(cls, num_channels):
        if num_channels ==1:
            return ('face')
        elif num_channels == 2:
            return ('left_eye', 'right_eye')
        elif num_channels == 3:
            return ('face', 'left_eye', 'right_eye')
        elif num_channels == 4:
            return ('face', 'left_eye', 'right_eye', 'face_grid')
        else:
            raise Exception("Unexpected input length {}".format(num_channels))

    def attribute_features(self, inputs, **kwargs):
        self.model.zero_grad()
        attributions = self.interpretation_algorithm.attribute(inputs, **kwargs)
        return attributions

    def interpretation_routine(self, inputs, input_names, label, sample_id, participant_id, window_index, window_size, overlap, input_dilation, data_path, attributions_path, interpretation_path, aus_data_path, generate_video_visualizations=False, generate_outstanding_frames_grid=False, generate_action_unit_cross_correlation=False, generate_region_importances=False):
        """
        The routine:
            - Creates a directory (A) for the current participant if it does not exist
            - Creates a directory (B) for the current sample_id under the participant's id
            - Runs the inputs through the interpretation algorithms: Saliency maps, Integrated Gradients, Integrated Gradients with noise tunnel, Deep Lift.
            - Saves the corresponding interpretation attribution to a directory named as the algorithm under B.
            - Keeps a log computed attribution deltas, labels and predicted outputs. 
        """
        #create output directories for attributions and for interpretations if it does not exist
        os.makedirs(interpretation_path, exist_ok=True)
        print('participant id: ', participant_id)
        print('sample id: ', sample_id)
        #create directory for participant if it does not exist
        participant_attr_dir = os.path.join(attributions_path, str(participant_id))
        participant_interpretation_dir = os.path.join(interpretation_path, str(participant_id))
        os.makedirs(participant_attr_dir, exist_ok=True)
        os.makedirs(participant_interpretation_dir, exist_ok=True)
        #create directory for current sample
        sample_attr_dir = os.path.join(participant_attr_dir, str(sample_id))
        sample_interpretation_dir = os.path.join(participant_interpretation_dir, str(sample_id))
        os.makedirs(sample_attr_dir, exist_ok = True)
        os.makedirs(sample_interpretation_dir, exist_ok = True)
        #get data directory
        participant_data_dir = os.path.join(data_path, str(participant_id))


        #directory path to save/load attributions to/from
        algorithm_attr_dir = os.path.join(sample_attr_dir, self.algorithm_name)
        #directory path to saveinterpretation to 
        algorithm_interpretation_dir = os.path.join(sample_interpretation_dir, self.algorithm_name)
        os.makedirs(algorithm_attr_dir, exist_ok = True)
        os.makedirs(algorithm_interpretation_dir, exist_ok = True)
        attributions_file_path = os.path.join(algorithm_attr_dir, 'attributions.pth.tar')
        
        #ATTRIBUTION COMPUTATION
        #compute attributions if flag is passed or if attributions file doesn't exist
        if self.compute_attributions:
            print("COMPUTING ATTRIBUTIONS!")
            #pass in extra arguments depending on the algorithm
            extra_args = {}
            if self.algorithm_name == 'NoiseTunnel':
                extra_args.update({'nt_type': 'smoothgrad_sq', 'n_samples': 2, 'stdevs': 1}) #1e-7, 0.1
            if self.interpretation_algorithm.has_convergence_delta():
                extra_args['return_convergence_delta'] = True
                #compute attributions
                attributions, delta = self.attribute_features(inputs, **extra_args)
                print("Delta: ", delta)
                attributions_obj = {'attributions': attributions, 'delta': delta}
            else:
                attributions = self.attribute_features(inputs, **extra_args)
                attributions_obj = {'attributions': attributions}
            #save attributions
            torch.save(attributions_obj, attributions_file_path)
        elif not self.compute_attributions and os.path.isfile(attributions_file_path):
            print("LOADING ATTRIBUTIONS! from path: ", attributions_file_path)
            attributions_obj = torch.load(attributions_file_path, map_location='cpu')
            attributions = attributions_obj['attributions']
        else:#don't compute attributions but there is no attributions file...
            warn(f"Attributions file filepath not found: {attributions_file_path}")
            warn("Skiping this sample without computing the attribution!")
            #do nothing, return
            return

        print("attributions sum: ", attributions.sum())
        #move computed attributions to numpy on cpu
        inputs = [input_channel.squeeze().cpu().detach().numpy() for input_channel in inputs]
        attributions = np.array([attribution.squeeze().cpu().detach().numpy() for attribution in attributions])
        #generate accumulated attributions signals for each input channel
        input_channel_attributions = attributions.sum(axis=(3,4), keepdims=False).sum(axis=1, keepdims=False)
        #generate accumulated attribution signal per frame
        framewise_attributions = input_channel_attributions.sum(axis=0, keepdims=True)
        #spawn writer processes jobs to handle the visualizations of the attributions input
        jobs = []


        #INTERPRETATION VIDEO GENERATION
        if generate_video_visualizations:
            for input_channel, input_name, attribution in zip(inputs, input_names, attributions):
                input_name_dir = os.path.join(algorithm_interpretation_dir, input_name)
                os.makedirs(input_name_dir, exist_ok=True)
                print("GENERATING VIDEO VISUALIZATIONS")
                #create interpretation videos for each input in inputs/original (tuple)
                for visualization_method in self.visualization_methods:
                    for sign in self.signs:
                        #make video destination path
                        video_name = '{}_{}.avi'.format(visualization_method, sign) if visualization_method != "original_video" else 'original_video.avi'
                        video_path = os.path.join(input_name_dir, video_name)
                        #create worker process
                        process = multiprocessing.Process(
                                target=self.video_writer_worker, 
                                args=(video_path, input_channel, input_name, attribution, visualization_method, sign)
                        )
                        jobs.append(process)
                        process.start()
                        if visualization_method == "original_video":
                            #no need to generate visualization more than once
                            break
            

        #INTERPRETATION OUTSTANDING IMAGE FRAMES GENERATION
        if generate_outstanding_frames_grid:
            print("GENERATING OUTSTANDING FRAMES GRID")
            #get the N largest/smallest frames
            largest_frame_indices = np.argpartition(np.squeeze(framewise_attributions), -self.N)[-self.N:]
            smallest_frame_indices= np.argpartition(np.squeeze(framewise_attributions), self.N)[: self.N]
            #visualize for each input channel
            for input_channel, input_name, attribution in zip(inputs, input_names, attributions):
                for visualization_method in self.visualization_methods:
                    for sign in self.signs:
                        #visualize TOP-K grid
                        print("Generating TOP/Bottom-K grid for {} {} {}".format(input_name, visualization_method, sign))
                        top_grid_dir = os.path.join(algorithm_interpretation_dir, 'top_{}_{}_grid'.format(self.N, input_name))
                        bottom_grid_dir = os.path.join(algorithm_interpretation_dir,'bottom_{}_{}_grid'.format(self.N, input_name))
                        os.makedirs(top_grid_dir, exist_ok=True)
                        os.makedirs(bottom_grid_dir, exist_ok=True)
                        image_grid_name = "{}_{}".format(visualization_method, sign) if visualization_method != "original_image" else "original_image"
                        #create image visualization and writer processes
                        process_top = multiprocessing.Process(
                                target = self.image_grid_writer_worker,
                                args=(
                                    os.path.join(top_grid_dir, image_grid_name),
                                    largest_frame_indices,
                                    input_channel, 
                                    attribution, 
                                    visualization_method, 
                                    sign
                                )
                        )
                        process_bottom = multiprocessing.Process(
                                target = self.image_grid_writer_worker, 
                                args=(
                                    os.path.join(bottom_grid_dir, image_grid_name), 
                                    smallest_frame_indices,
                                    input_channel, 
                                    attribution, 
                                    visualization_method, 
                                    sign
                                )
                        )
                        #start and track the processes
                        jobs.append(process_top); jobs.append(process_bottom)
                        process_top.start(); process_bottom.start()
                        if visualization_method == "original_image":
                            #then there is no need to visualize for multiple signs!
                            break


        #INTERPRETATION WITH ACTION UNITS (AUs) correlation
        if generate_action_unit_cross_correlation:
            print("GENERATING AU CROSS CORRELATIONS")
            #create directory for region analysis
            au_correlations_dir = os.path.join(algorithm_interpretation_dir, 'au_correlations')
            os.makedirs(au_correlations_dir, exist_ok=True)
            #load data
            participant_au = load_participant_AU(aus_data_path, participant_id)
            is_valid_frames = np.load(os.path.join(participant_data_dir, 'is_valid_frames.npy'))
            sample_au = get_sample_AUS(participant_au, is_valid_frames, window_size, window_index, overlap, input_dilation)
            #remove all columns but AU measures
            effective_aus = get_effective_AUS(sample_au)
            #add attributions to dataframe
            effective_aus['framewise_attributions'] = framewise_attributions.flatten()
            #store the sample's effective AUs
            effective_aus.to_csv(os.path.join(au_correlations_dir, 'aus_summary.csv'))
            #compute correlations
            correlation_dataframe = effective_aus.corr(method='spearman')
            attribution_correlations = correlation_dataframe['framewise_attributions']
            #drop the trivial framewiseattribution
            attribution_correlations = attribution_correlations.drop('framewise_attributions')
            effective_aus = effective_aus.drop(columns='framewise_attributions')
            # save dataframe to csv
            correlation_dataframe.to_csv(os.path.join(au_correlations_dir, 'all_correlations.csv'))
            #plot correlations
            plt.figure(figsize=(20,16))
            sns.heatmap(correlation_dataframe, annot=True)
            plt.savefig(os.path.join(au_correlations_dir, 'all_correlations.png'))
            #plt.show()
            plt.close()
            # now for just attribution correlations
            # NOTE: filter out AU_c. only use AU_r intensities
            au_types = ['c', 'r']#recognition and intensity
            for au_type in au_types:
                #store effective type attribution
                effective_aus_subtype = effective_aus.filter(regex=au_type)
                effective_aus_subtype.to_csv(os.path.join(au_correlations_dir, f"aus_{au_type}_summary.csv"))
                #sum columns and just a bar plot of resulting
                effective_aus_subtype_histogram = effective_aus_subtype.sum()
                #plot histogram of au subtype
                effective_aus_subtype_histogram.plot.barh()
                plt.savefig(os.path.join(au_correlations_dir, f"aus_{au_type}_histogram.png"))
                plt.close()
                attribution_correlations_type = attribution_correlations.filter(regex=au_type)
                attribution_correlations_type.to_csv(os.path.join(au_correlations_dir, f'aus_{au_type}_attribution_correlations.csv'))
                plt.figure()
                #sns.heatmap([attribution_correlations_type], annot=True)
                attribution_correlations_type.plot.barh()
                plt.savefig(os.path.join(au_correlations_dir, f"aus_{au_type}_attribution_correlations.png"))
                #plt.show()
                plt.close()


        #GENERATE REGION IMPORTANCE ATTRIBUTIONS
        if generate_region_importances:
            print("GENERATING IMPORTANCE REGIONS")
            #create directory for region analysis
            region_importances_dir = os.path.join(algorithm_interpretation_dir, 'region_importances')
            os.makedirs(region_importances_dir, exist_ok=True)
            #load is_valid_frames, landmark grids, face_grids
            is_valid_frames = np.load(os.path.join(participant_data_dir, 'is_valid_frames.npy'))
            sample_frames = get_frame_indices(window_size, window_index, overlap, input_dilation)
            face_grids = np.load(os.path.join(participant_data_dir, 'face_grids.npy'))
            landmark_grids = np.load(os.path.join(participant_data_dir, 'landmark_grids.npy'))
            for attribution, _input, input_name in zip(attributions, inputs, input_names):
                #only works on face input
                if input_name != 'face':
                    continue
                #initialize placeholder
                region_temporal_attributions = {}
                #project landmarks (of original video) to dilated face crop
                dilated_landmarks_grid = self.generate_dilated_landmarks(attribution, landmark_grids, face_grids, is_valid_frames, sample_frames)
                for frame, frame_attributions, dilated_landmarks in zip(np.transpose(_input, (1, 2, 3, 0)), np.transpose(attribution, (1, 2, 3, 0)), dilated_landmarks_grid):
                    #NOTE: uncomment for visualization of dilated landmarks
                    """
                    frame = np.transpose((frame / 2) + 0.5, (1,2,0))
                    plt.imshow(frame)
                    plt.scatter(dilated_landmarks[:, 0], dilated_landmarks[:,1], 2)
                    plt.show()
                    """
                    for roi_name in FACIAL_LANDMARKS_IDXS.keys():
                        if roi_name == 'jaw':
                            #skip jaw
                            continue
                        roi = get_roi(frame, dilated_landmarks, roi_name, padding=0.25, padding_mode='length')
                        attribution_roi = get_roi(frame_attributions, dilated_landmarks, roi_name, padding=0.25, padding_mode='length')
                        #add region attribution score to temporal attributions
                        region_attribution_score = attribution_roi.sum()
                        region_temporal_attributions[roi_name] = np.append(region_temporal_attributions.get(roi_name, np.array([])), region_attribution_score)
                        #include frame attribution score
                for data_modality in ['raw', 'normalized']:
                    #save temporal region attributions
                    region_temporal_attributions_df = pd.DataFrame.from_dict(region_temporal_attributions)
                    #summarize temporal information into a single score
                    regionwise_attributions_df = region_temporal_attributions_df.sum()
                    if data_modality == 'normalized':
                        #normalize by total sum of attributions
                        regionwise_attributions_df = regionwise_attributions_df / attributions.sum()
                    #save dataframes
                    region_temporal_attributions_df.to_csv(os.path.join(region_importances_dir, f'{data_modality}_regionwise_temporal_attributions.csv'))
                    regionwise_attributions_df.to_csv(os.path.join(region_importances_dir, f'{data_modality}_regionwise_attributions.csv'))
                    #generate and save plots
                    region_temporal_attributions_df.plot(title=f"{data_modality} temporal region importances")
                    plt.savefig(os.path.join(region_importances_dir, f'{data_modality}_regionwise_temporal_attributions.png'))
                    plt.close()
                    #regionwise_attributions_df.plot(kind='barh', figsize=(12,12), title = f"{data_modality} region importances")
                    regionwise_attributions_df.plot(kind='barh', title = f"{data_modality} region importances")
                    plt.tight_layout()
                    plt.savefig(os.path.join(region_importances_dir, f'{data_modality}_regionwise_attributions.png'))
                    plt.close()
                

                

            #wait for all processes to complete before returning
            for running_process in jobs:
                running_process.join()


    @classmethod
    def find_cluster(cls, label):
        for cluster_name, cluster_range in cls.CLUSTERS.items():
            if cluster_range[0] <= label <= cluster_range[1]:
                return cluster_name
        raise Exception(f"No cluster was found for {label}")


    @classmethod
    def generate_dilated_landmarks(cls, attribution, landmark_grids, face_grids, is_valid_frames=None, sample_frames=None):
        """
        Generates scores for each face region from: 
        @param attributions: the attributions for the model input
        @param landmark_grids: the landmarks for each frame in the model input
        @param face_grids: represents an array of face rectangles (x, y, w, h)
        @param is_valid_frames (None or [bool] or [1 or 0]): defines input positions of valid and invalid frames to filter out from landmark_grids and face_grids if present. 
        @param sample_frames (None or [int]): defines the indexes of frames to sample from landmark_grids and face_grids when defining the region of importance
        """
        if is_valid_frames is not None:
            #filter for only valid grids and valid landmarks
            face_grids = face_grids[is_valid_frames]
            landmark_grids = landmark_grids[is_valid_frames]
        if sample_frames is not None:
            #slice for sample's frames
            face_grids = face_grids[sample_frames]
            landmark_grids = landmark_grids[sample_frames]
        #concatenate attributions to single channel
        attribution = attribution.sum(axis=0)
        assert len(attribution) == len(face_grids) == len(landmark_grids), "Lengths of attribution, face_grids and landmark_grids must be equal, but ({}, {}, {})".format(len(attribution), len(face_grids), len(landmark_grids))
        #center landmarks relative to face rectangle
        translation_centering = face_grids[:, np.newaxis, :2]
        landmark_grids = landmark_grids - translation_centering
        #get dilation factors
        input_dimensions = attribution.shape[-2:]
        #dilate landmarks to scaled input
        face_grid_dimmensions = face_grids[:, np.newaxis, 2:]
        dilation_factor = input_dimensions / face_grid_dimmensions
        landmark_grids = landmark_grids * (dilation_factor)
        #round to obtain back pixels
        return np.rint(landmark_grids).astype(np.float32)

    
    @classmethod
    def image_grid_writer_worker(cls, image_path, indices, input_channel, attribution, visualization_method, sign):
        #ForkedPdb().set_trace()
        grid, _ = viz.visualize_multiple_image_attr(
                np.transpose(attribution, (1, 2, 3, 0))[indices], 
                np.transpose(input_channel,(1, 2, 3, 0))[indices], 
                [visualization_method] * cls.N, 
                [sign] * cls.N
        )
        grid.savefig(image_path)


    @classmethod
    def video_writer_worker(cls, video_path, input_channel, input_name, attribution, visualization_method, sign):
        #initialize video writer
        video_writer = animation.FFMpegWriter(fps=cls.FRAMES_PER_SECOND)
        cmap = LinearSegmentedColormap.from_list(
                name='Custom detailed map',
                colors=[
                    (0, '#000000'),#black for lowest attr
                    (0.2, '#0000ff'),#blue for half negative attr
                    (0.5, '#ffffff'),#white for neutral attribution
                    (0.7,'#ffff00'),#yellowfor half positive attr
                    (1, '#ff0000') #red for fully positive attr
                ]
        )
        interpretation_video = viz.visualize_video_attribution(input_channel, cls.FRAMES_PER_SECOND, attribution, method=visualization_method, sign=sign, cmap=cmap)
        interpretation_video.save(video_path, writer=video_writer)
        plt.close()#Close figure for current thread


    @classmethod
    def generate_interpretation_video(cls, video, video_attributions, method = "blended_heat_map", sign="absolute_value"):
        """
        Generates an interpretation video combining the original input with the passed in attribution using combine heatmaps
        cls: classname inherited from the class
        input_batch: requires batch_size = 1
        attributions: the computed significance attributions computed for the corresponding input
        method (str): the visualization method to use: heatmap, blended_heat_map, ...
        sign (str): selects the sign of attributions to display positive, negative, all or defaults to absolute_value
        """
        #initialize video output frames
        interpretation_frames = []
        #video_writter = animation.FFMpegWriter(fps=FRAMES_PER_SECOND)
        fig = plt.figure()
        ax = fig.add_subplot()
        for image, image_attributions in zip(video, video_attributions):
            #generate interpretation for the current image frame
            #visualize without transpose
            image = np.transpose((image / 2) + 0.5, (1,2,0))
            ### NOTE: visualize_image_attr was modified from Captum to return also the heatmap object with this call
            fig, ax, axes_frame_images= viz.visualize_image_attr(image_attributions, image, method=method, sign=sign, show_colorbar=False, use_pyplot=False, plt_fig_axis=(fig, ax))
            interpretation_frames.append(axes_frame_images)
        #save frames into a video object
        video_animation = animation.ArtistAnimation(fig, interpretation_frames, interval=cls.FRAMES_PER_SECOND)
        return video_animation

    
    @classmethod 
    def plot_attributions(cls, attributions, directory_path, plot_name):
        num_channels, num_frames = attributions.shape
        x_axis = np.arange(0, num_frames)
        x_matrix = np.array([x_axis,] * num_channels)
        plt.plot(x_matrix.T, attributions.T)
        plt.legend(cls.get_input_names(num_channels), loc='upper center', ncol=3, bbox_to_anchor=[0.5, 1.1], columnspacing=1.0, labelspacing=0.0, handletextpad=0.0, handlelength=1.5,)
        plot_path = os.path.join(directory_path, plot_name)
        plt.savefig(plot_path)
        plt.close()



