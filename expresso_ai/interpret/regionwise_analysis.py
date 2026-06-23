import os

import matplotlib.pyplot as plt
import pandas as pd

from .cluster import generate_cluster_plot, find_cluster, determine_prediction_cluster, generate_regional_prediction_plot


def generate_participant_region_summary(interpretation_path, algorithm_name, data_type):
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            continue
        #initialize participant summary
        participant_region_accumulator = pd.DataFrame()
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            regions_dir = os.path.join(sample_dir, algorithm_name, 'region_importances')
            #read region-wise attributions
            regionwise_attributions = pd.read_csv(os.path.join(regions_dir,f'{data_type}_regionwise_attributions.csv'), index_col=0)
            #accumulate to participant
            participant_region_accumulator = pd.concat([participant_region_accumulator, regionwise_attributions], axis=1)
        #compute mean 
        participant_region_accumulator = participant_region_accumulator.mean(axis=1)
        #save participant mean
        summaries_dir = os.path.join(participant_dir, algorithm_name, 'region_importances')
        os.makedirs(summaries_dir, exist_ok=True)
        participant_region_accumulator.to_csv(os.path.join(summaries_dir, f'{data_type}_regionwise_attributions_summary.csv'))
        #and save plot
        participant_region_accumulator.plot(kind='barh')
        plt.savefig(os.path.join(summaries_dir, f'{data_type}_regionwise_attributions_summary.png'))
        plt.close()


def generate_prediction_clusters(interpretation_path, algorithm_name, cluster_ranges, correct_treshold, incorrect_treshold, decision_type, data_type):
    #initialize accumualators dir
    accumulators_dir = os.path.join(interpretation_path, algorithm_name, 'region_importances')
    os.makedirs(accumulators_dir, exist_ok=True)
    #read labels assigned to samples
    labels_df = pd.read_csv(os.path.join(interpretation_path, 'interpretation_log.csv'), index_col=0)
    # Generate cluster representatives for region attributions
    cluster_accumulators = {cluster_name: pd.DataFrame() for cluster_name in cluster_ranges}
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            print(f"could not open participant dir {participant_dir}. Continuing")
            continue
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            regions_dir = os.path.join(sample_dir, algorithm_name, 'region_importances')
            #read region-wise attributions
            regionwise_attributions = pd.read_csv(os.path.join(regions_dir,f'{data_type}_regionwise_attributions.csv'), index_col=0)
            #accumulate to representative cluster
            sample_row = labels_df[labels_df['sample_id'] == sample_id]
            sample_prediction = int(sample_row['prediction'])
            sample_label = int(sample_row['label'])
            #compute error
            error = sample_label - sample_prediction
            if decision_type == 'correct':
                #assert that error is within allowed treshold
                if error < (-1) * correct_treshold or correct_treshold < error:
                    #incorrectly labeled
                    continue
            elif decision_type == 'underestimated':
                #assert that error is larter than treshold. otherwise continue
                if error <= incorrect_treshold:
                    continue
            elif decision_type == 'overestimated':
                #assert that error is smaller than negative treshold. otherwise continue
                if error >= (-1) * incorrect_treshold:
                    continue
            elif decision_type == 'overall':
                #no assess. add to cluster
                pass
            else:
                raise NameError(f"{decision_type} is not valid")
            #
            cluster_name = find_cluster(sample_label, cluster_ranges)
            #update cluster accumulator
            cluster_accumulators[cluster_name] = pd.concat(
                    [
                        cluster_accumulators[cluster_name], 
                        regionwise_attributions], 
            axis = 1)
    #save clusters
    save_cluster(cluster_accumulators, accumulators_dir, decision_type=decision_type)


def save_cluster(cluster_accumulators, accumulators_dir, decision_type):

    cluster_means = {}
    for cluster_name in cluster_accumulators.keys():
        try:
            #compute mean
            cluster_mean = cluster_accumulators[cluster_name].mean(axis=1)
            cluster_means[cluster_name]  = cluster_mean
            #save attributions
            print('accumulators dir: ', accumulators_dir)
            cluster_mean.to_csv(os.path.join(accumulators_dir, f'{decision_type}_{cluster_name}_attributions.csv'))
        except:
            #cluster accumulators may be empty. continue
            print(f"Could not generate summary cluster for {cluster_name}")
            pass
    #generate plots
    generate_cluster_plot(cluster_means, decision_type, output_dir=accumulators_dir)


def generate_regional_prediction_clusters(interpretation_path, algorithm_name, cluster_of_interest, cluster_ranges, correct_treshold, incorrect_treshold, plot_name='region-wise'):
    PREDICTION_CLUSTERS = ('overestimated', 'underestimated', 'correct', 'overall')
    #cluster tresholds
    cluster_tresholds = cluster_ranges[cluster_of_interest]
    #initialize accumualators dir
    accumulators_dir = os.path.join(interpretation_path, algorithm_name, plot_name)
    os.makedirs(accumulators_dir, exist_ok=True)
    #read labels assigned to samples
    labels_df = pd.read_csv(os.path.join(interpretation_path, 'interpretation_log.csv'), index_col=0)
    # Generate cluster representatives for region attributions
    cluster_accumulators = {cluster_name: pd.DataFrame() for cluster_name in PREDICTION_CLUSTERS}
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            continue
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            regions_dir = os.path.join(sample_dir, algorithm_name, 'region_importances')
            #read region-wise attributions
            regionwise_attributions = pd.read_csv(os.path.join(regions_dir,'regionwise_attributions.csv'), index_col=0)
            #accumulate to representative cluster
            sample_row = labels_df[labels_df['sample_id'] == sample_id]
            sample_prediction = int(sample_row['prediction'])
            sample_label = int(sample_row['label'])
            #see if sample fits in cluster
            if sample_prediction < cluster_tresholds[0] or cluster_tresholds[1] < sample_prediction:
                #sample is outside cluster of interest
                continue
            #compute error:
            error = sample_label - sample_prediction
            cluster_accumulators['overall'] = pd.concat(
                    [
                        cluster_accumulators['overall'],
                        regionwise_attributions
                    ],
            axis = 1)
            prediction_cluster_name = determine_prediction_cluster(error, correct_treshold, incorrect_treshold)
            if prediction_cluster_name is None:
                print(f"error {error} belongs to no cluster")
                continue
            #update cluster accumulator
            cluster_accumulators[prediction_cluster_name] = pd.concat(
                    [
                        cluster_accumulators[prediction_cluster_name], 
                        regionwise_attributions], 
            axis = 1)
    #save clusters
    prefix = cluster_of_interest
    cluster_means = {}
    for cluster_name in PREDICTION_CLUSTERS:
        #compute mean
        cluster_mean = cluster_accumulators[cluster_name].mean(axis=1)
        #if len(cluster_mean) != 0:
        cluster_means[cluster_name]  = cluster_mean
        #save attributions
        cluster_mean.to_csv(os.path.join(accumulators_dir, f'{prefix}_{cluster_name}_attributions.csv'))
    #generate plots
    generate_regional_prediction_plot(cluster_means, prefix, output_dir=accumulators_dir)
