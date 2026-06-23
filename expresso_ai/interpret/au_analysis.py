import os

import matplotlib.pyplot as plt
import pandas as pd

from .cluster import generate_cluster_plot, find_cluster


def generate_participant_AU_summary(interpretation_path, algorithm_name, class_name='c'):
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            continue
        print('participant_dir: ', participant_dir)
        #initialize participant summary
        participant_AU_accumulator = pd.DataFrame()
        participant_AU_corr_accumulator = pd.DataFrame()
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            print('sample_dir: ', sample_dir)
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            algorithm_dir = os.path.join(sample_dir, algorithm_name)
            #read AUs
            au_scores = pd.read_csv(os.path.join(algorithm_dir, 'au_correlations', f'aus_{class_name}_summary.csv'), index_col=0)
            #read AU-corr attributions
            au_correlations = pd.read_csv(os.path.join(algorithm_dir, 'au_correlations', f'aus_{class_name}_attribution_correlations.csv'), index_col=0)
            #accumulate to participant
            participant_AU_accumulator = pd.concat([participant_AU_accumulator, au_scores], axis=0)
            participant_AU_corr_accumulator = pd.concat([participant_AU_corr_accumulator, au_correlations], axis=1)
        #compute mean 
        #import pdb; pdb.set_trace()
        participant_AU_accumulator = participant_AU_accumulator.mean(axis=0)
        participant_AU_corr_accumulator = participant_AU_corr_accumulator.mean(axis=1)
        #save participant mean
        au_output_dir = os.path.join(participant_dir, algorithm_name, 'au_correlations')
        os.makedirs(au_output_dir, exist_ok=True)
        participant_AU_accumulator.to_csv(os.path.join(au_output_dir, f'au_{class_name}_summary.csv'))
        participant_AU_corr_accumulator.to_csv(os.path.join(au_output_dir, f'au_{class_name}_attribution_correlations.csv'))
        #and save plot
        participant_AU_accumulator.plot(kind='barh')
        plt.savefig(os.path.join(au_output_dir, f'effective_au_{class_name}_summary.png'))
        plt.close()
        participant_AU_corr_accumulator.plot(kind='barh')
        plt.savefig(os.path.join(au_output_dir, f'au_correlation_{class_name}_attributions.png'))
        plt.close()
        try:
            os.remove(os.path.join(au_output_dir, f'effective_au_{class_name}_attributions.png'))
        except:
            pass


def generate_overall_prediction_clusters(interpretation_path, algorithm_name, cluster_ranges, plot_name='region-wise', class_name='c'):
    #initialize accumualators dir
    accumulators_dir = os.path.join(interpretation_path, algorithm_name, plot_name)
    os.makedirs(accumulators_dir, exist_ok=True)
    #read labels assigned to samples
    labels_df = pd.read_csv(os.path.join(interpretation_path, 'interpretation_log.csv'), index_col=0)
    # Generate cluster representatives
    cluster_accumulators = {cluster_name: pd.DataFrame() for cluster_name in cluster_ranges}
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            continue
        print('participant_dir: ', participant_dir)
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            print('sample_dir: ', sample_dir)
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            algorithm_dir = os.path.join(sample_dir, algorithm_name)
            #read attributions
            au_correlations = pd.read_csv(os.path.join(algorithm_dir, 'au_correlations', f'aus_{class_name}_attribution_correlations.csv'), index_col=0)
            #accumulate to representative cluster
            sample_prediction = int(labels_df[labels_df['sample_id'] == sample_id]['prediction'])
            sample_label = int(labels_df[labels_df['sample_id'] == sample_id]['prediction'])
            cluster_name = find_cluster(sample_label, cluster_ranges)
            #update cluster accumulator
            cluster_accumulators[cluster_name] = pd.concat([
                cluster_accumulators[cluster_name], 
                au_correlations], 
            axis = 1)
    prefix = 'overall'
    #get mean of clusters
    cluster_means = {cluster_name: cluster.mean(axis=1) for cluster_name, cluster in cluster_accumulators.items()}
    #save clusters
    for cluster_name, cluster in cluster_means.items():
        cluster.to_csv(os.path.join(accumulators_dir, f'{prefix}_{cluster_name}_attribution_correlations.csv'))
    #generate plots
    generate_cluster_plot(cluster_means, prefix, output_dir=accumulators_dir, plot_name=plot_name)


def generate_correct_prediction_clusters(interpretation_path, algorithm_name, cluster_ranges, treshold, plot_name='region-wise', class_name='c'):
    #initialize accumualators dir
    accumulators_dir = os.path.join(interpretation_path, algorithm_name, plot_name)
    os.makedirs(accumulators_dir, exist_ok=True)
    #read labels assigned to samples
    labels_df = pd.read_csv(os.path.join(interpretation_path, 'interpretation_log.csv'), index_col=0)
    # Generate cluster representatives
    cluster_accumulators = {cluster_name: pd.DataFrame() for cluster_name in cluster_ranges}
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            continue
        print('participant_dir: ', participant_dir)
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            print('sample_dir: ', sample_dir)
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            algorithm_dir = os.path.join(sample_dir, algorithm_name)
            #read attributions
            au_correlations = pd.read_csv(os.path.join(algorithm_dir, 'au_correlations', f'aus_{class_name}_attribution_correlations.csv'), index_col=0)
            #accumulate to representative cluster
            sample_row = labels_df[labels_df['sample_id'] == sample_id]
            sample_prediction = int(sample_row['prediction'])
            sample_label = int(sample_row['label'])
            #compute error
            error = sample_label - sample_prediction
            if error < (-1) * treshold or treshold < error:
                #incorrectly labeled
                continue
            cluster_name = find_cluster(sample_label, cluster_ranges)
            #update cluster accumulator
            cluster_accumulators[cluster_name] = pd.concat(
                    [
                        cluster_accumulators[cluster_name], 
                        au_correlations], 
            axis = 1)
    #save clusters
    prefix = 'correct'
    cluster_means = {}
    for cluster_name in cluster_ranges:
        try:
            #compute mean
            cluster_mean = cluster_accumulators[cluster_name].mean(axis=1)
            cluster_means[cluster_name]  = cluster_mean
            #save attributions
            print('accumulators dir: ', accumulators_dir)
            cluster_mean.to_csv(os.path.join(accumulators_dir, f'{prefix}_{cluster_name}_attribution_correlations.csv'))
        except:
            #cluster accumulators may be empty. continue
            print(f"Could not generate summary cluster for {cluster_name}")
            pass
    #generate plots
    generate_cluster_plot(cluster_means, prefix, output_dir=accumulators_dir, plot_name=plot_name)


def generate_underestimated_prediction_clusters(interpretation_path, algorithm_name, cluster_ranges, treshold, plot_name='region-wise', class_name='c'):
    #initialize accumualators dir
    accumulators_dir = os.path.join(interpretation_path, algorithm_name, plot_name)
    os.makedirs(accumulators_dir, exist_ok=True)
    #read labels assigned to samples
    labels_df = pd.read_csv(os.path.join(interpretation_path, 'interpretation_log.csv'), index_col=0)
    # Generate cluster representatives
    cluster_accumulators = {cluster_name: pd.DataFrame() for cluster_name in cluster_ranges}
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            continue
        print('participant_dir: ', participant_dir)
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            print('sample_dir: ', sample_dir)
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            algorithm_dir = os.path.join(sample_dir, algorithm_name)
            #read attributions
            au_correlations = pd.read_csv(os.path.join(algorithm_dir, 'au_correlations', f'aus_{class_name}_attribution_correlations.csv'), index_col=0)
            #accumulate to representative cluster
            sample_row = labels_df[labels_df['sample_id'] == sample_id]
            sample_prediction = int(sample_row['prediction'])
            sample_label = int(sample_row['label'])
            #compute error
            error = sample_label - sample_prediction
            if error < treshold:
                continue
            cluster_name = find_cluster(sample_label, cluster_ranges)
            #update cluster accumulator
            cluster_accumulators[cluster_name] = pd.concat(
                    [
                        cluster_accumulators[cluster_name], 
                        au_correlations], 
            axis = 1)
    #save clusters
    prefix = 'underestimated'
    cluster_means = {}
    for cluster_name in cluster_ranges:
        #compute mean
        cluster_mean = cluster_accumulators[cluster_name].mean(axis=1)
        if len(cluster_mean) != 0:
            cluster_means[cluster_name]  = cluster_mean
            #save attributions
            print('accumulators dir: ', accumulators_dir)
            try:#TODO REMOVE TRY/EXCEPT
                os.remove(os.path.join(accumulators_dir, f'{prefix}_{cluster_name}_attribution_correlations.csv'))
            except:
                pass
            cluster_mean.to_csv(os.path.join(accumulators_dir, f'{prefix}_{cluster_name}_au_{class_name}_attribution_correlations.csv'))
    #generate plots
    generate_cluster_plot(cluster_means, prefix, output_dir=accumulators_dir, plot_name=plot_name)


def generate_overestimated_prediction_clusters(interpretation_path, algorithm_name, cluster_ranges, treshold, plot_name='region-wise', class_name='c'):
    #initialize accumualators dir
    accumulators_dir = os.path.join(interpretation_path, algorithm_name, plot_name)
    os.makedirs(accumulators_dir, exist_ok=True)
    #read labels assigned to samples
    labels_df = pd.read_csv(os.path.join(interpretation_path, 'interpretation_log.csv'), index_col=0)
    # Generate cluster representatives
    cluster_accumulators = {cluster_name: pd.DataFrame() for cluster_name in cluster_ranges}
    #iterate over participants to generate participant-wise clusters
    participant_paths = [os.path.join(interpretation_path, f) for f in os.listdir(interpretation_path) if os.path.isdir(os.path.join(interpretation_path,f))]
    for participant_dir in participant_paths:
        try:
            participant_id = int(os.path.basename(participant_dir))
        except:
            continue
        print('participant_dir: ', participant_dir)
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            print('sample_dir: ', sample_dir)
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            algorithm_dir = os.path.join(sample_dir, algorithm_name)
            #read attributions
            au_correlations = pd.read_csv(os.path.join(algorithm_dir, 'au_correlations', f'aus_{class_name}_attribution_correlations.csv'), index_col=0)
            #accumulate to representative cluster
            sample_row = labels_df[labels_df['sample_id'] == sample_id]
            sample_prediction = int(sample_row['prediction'])
            sample_label = int(sample_row['label'])
            #compute error
            error = sample_label - sample_prediction
            if error >= (-1) * treshold:
                #not overestimated. Continue
                continue
            cluster_name = find_cluster(sample_label, cluster_ranges)
            #update cluster accumulator
            cluster_accumulators[cluster_name] = pd.concat(
                    [
                        cluster_accumulators[cluster_name], 
                        au_correlations], 
            axis = 1)
    #save clusters
    prefix = 'overestimated'
    cluster_means = {}
    for cluster_name in cluster_ranges:
        #compute mean
        cluster_mean = cluster_accumulators[cluster_name].mean(axis=1)
        if len(cluster_mean) != 0:
            cluster_means[cluster_name]  = cluster_mean
            #save attributions
            print('accumulators dir: ', accumulators_dir)
            cluster_mean.to_csv(os.path.join(accumulators_dir, f'{prefix}_{cluster_name}_attribution_correlations.csv'))
    #generate plots
    generate_cluster_plot(cluster_means, prefix, output_dir=accumulators_dir, plot_name=plot_name)


def generate_regional_prediction_clusters(interpretation_path, algorithm_name, cluster_ranges, cluster_of_interest, correct_treshold, incorrect_treshold, plot_name='region-wise'):
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
        print('participant_dir: ', participant_dir)
        sample_dirs  = [os.path.join(participant_dir, f) for f in os.listdir(participant_dir) if os.path.isdir(os.path.join(participant_dir,f))]
        for sample_dir in sample_dirs:
            print('sample_dir: ', sample_dir)
            try:
                sample_id = int(os.path.basename(sample_dir))
            except:
                print(f"could not open sample dir {sample_dir}. Continuing")
                continue
            algorithm_dir = os.path.join(sample_dir, algorithm_name)
            #read region-wise attributions
            au_correlations = pd.read_csv(os.path.join(algorithm_dir,f'au_{class_name}_attribution_correlations.csv'), index_col=0)
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
                        au_correlations
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
                        au_correlations], 
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
        cluster_mean.to_csv(os.path.join(accumulators_dir, f'{prefix}_{cluster_name}_attribution_correlations.csv'))
    #generate plots
    generate_regional_prediction_plot(cluster_means, prefix, output_dir=accumulators_dir)
