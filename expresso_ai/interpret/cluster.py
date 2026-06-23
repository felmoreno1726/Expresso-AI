import os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np


def find_cluster(label, cluster_ranges):
    for cluster_name, cluster_range in cluster_ranges.items():
        if cluster_range[0] <= label <= cluster_range[1]:
            return cluster_name
    raise Exception(f"No cluster was found for {label}")


def generate_cluster_plots(clusters, prefix, output_dir, plot_name='region-wise'):
    fig, axes = plt.subplots(len(clusters), 1, figsize=(12, 24))
    fig.suptitle(f"{prefix} plot_name clusters")
    for i, (cluster_name, cluster) in enumerate(clusters.items()):
        axis = axes[i]
        regions = cluster.keys()
        scores = cluster.values
        y_pos = np.arange(len(regions))
        #axis.barh(y_pos, scores, align='center', color=plt.cm.Set1(np.arange(len(regions))), edgecolor=np.zeros(len(regions)))
        axis.barh(y_pos, scores, align='center', color=plt.cm.Set1(np.arange(len(regions))), edgecolor='black')
        #axis.barh(y_pos, scores, align='center', color=plt.cm.Set1(np.arange(len(regions))))
        axis.set_yticks(y_pos)
        axis.set_yticklabels(regions)
        axis.invert_yaxis()
        #axis.set_xlabel("attributions")
        axis.set_ylabel(cluster_name)
        #plt.show()
    plt.xlabel('attributions')
    plt.savefig(os.path.join(output_dir, f'{prefix}_{plot_name}_attributions.png'))
    plt.close()

def generate_cluster_plot(clusters, prefix, output_dir, plot_name='region-wise'):
    fig, axis = plt.subplots(figsize=(12, 12))
    fig.suptitle(f"{prefix} {plot_name} clusters")
    pivoted_clusters = pivot_cluster(clusters)
    barwidth = 1.0 / (len(pivoted_clusters) + 1)
    #initialize height
    max_y = 0
    min_y = 0
    severities = []
    #colors = plt.cm.gist_rainbow(np.arange(len(pivoted_clusters)))
    num_colors = len(pivoted_clusters)
    print('num colors: ', num_colors)
    #cm = plt.cm.gist_rainbow
    cm = plt.get_cmap('gist_rainbow')
    #cNorm = colors.Normalize(vmin=0, vmax=num_colors-1)
    #scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    #axis.set_prop_cycle([scalarMap.to_rgba(i) for i in range(num_colors)])
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]
    axis.set_prop_cycle(color=colors)
    for i, (column_name, cluster) in enumerate(pivoted_clusters.items()):
        #axis = axes[i]
        severities = cluster.keys()
        scores = cluster.values()
        max_y = max(max_y, max(scores))
        min_y = min(min_y, min(scores))
        x_pos = np.arange(len(severities)) + i * barwidth
        #axis.bar(x_pos, scores, align='center', width=barwidth, color=colors[i], edgecolor='black', label=column_name)
        axis.bar(x_pos, scores, align='center', width=barwidth, edgecolor='black', label=column_name)
        #axis.set_yticks(x_pos)
        #plt.show()
    #x labels to the middle of bar groups
    plt.xticks([r + barwidth * (len(severities) / 2) for r in range(len(severities))], severities)
    plt.yticks([min_y, max_y])
    plt.ylim(min_y*1.5, max_y*1.5)
    plt.ylabel('attributions')
    plt.legend(loc='upper left', ncol=num_colors//4)
    #plt.show()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{plot_name}_attributions.png'))
    plt.close()


def pivot_cluster(clusters):
    plot_columns = list(clusters.keys())
    plot_names = get_subkeys(clusters)
    pivoted_clusters = {}
    for plot_name in plot_names:
        plot= {}
        for column in plot_columns:
            #add column to plot
            try:
                plot[column] = clusters[column][plot_name]
            except:
                plot[column] = 0
        pivoted_clusters[plot_name] = plot
    return pivoted_clusters

def get_subkeys(dic):
    subkeys = set()
    for key in dic.keys():
        print('key: ', key)
        subkeys.update(dic[key].keys())
        #print('subkeys: ', dic[key])
    return subkeys

def generate_regional_prediction_plot(clusters, prefix, output_dir, plot_name='region-wise'):
    fig, axis = plt.subplots(figsize=(12, 12))
    fig.suptitle(f"{prefix} {plot_name} clusters")
    #pivoted_clusters = pivot_cluster(clusters)
    pivoted_clusters = clusters
    clusters = pivot_cluster(clusters)
    barwidth = 1.0 / (len(pivoted_clusters) + 1)
    #initialize height
    max_y = 0
    min_y = 0
    colors = plt.cm.Set1(np.arange(len(pivoted_clusters)))
    for i, (column_name, cluster) in enumerate(pivoted_clusters.items()):
        #axis = axes[i]
        severities = cluster.keys()
        scores = cluster.values
        max_y = max(max_y, max(scores, default=0))
        min_y = min(min_y, min(scores, default=0))
        x_pos = np.arange(len(severities)) + i * barwidth
        #axis.bar(x_pos, scores, align='center', width=barwidth, color=plt.cm.Set1(np.arange(len(severities))), edgecolor='black', label=column_name)
        axis.bar(x_pos, scores, align='center', width=barwidth, color=colors[i], edgecolor='black', label=column_name)
        axis.set_yticks(x_pos)
        #plt.show()
    #x labels to the middle of bar groups
    plt.xticks([r + barwidth * (len(pivoted_clusters) / 2) for r in range(len(severities))], severities)
    plt.ylim(min_y*1.5, max_y*1.5)
    plt.ylabel('attributions')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{plot_name}_attributions.png'))
    #plt.show()
    plt.close()

def generate_regional_prediction_plots(clusters, prefix, output_dir, plot_name='region-wise'):
    #pivot clusters
    clusters = pivot_cluster(clusters)
    fig, axes = plt.subplots(len(clusters), 1, figsize=(12, 24))
    fig.suptitle(f"{prefix} {plot_name} clusters")
    for i, (cluster_name, cluster) in enumerate(clusters.items()):
        axis = axes[i]
        regions = cluster.keys()
        scores = cluster.values()
        y_pos = np.arange(len(regions))

        axis.barh(y_pos, scores, align='center', color=plt.cm.Set1(np.arange(len(regions))), edgecolor='black')
        axis.set_yticks(y_pos)
        axis.set_yticklabels(regions)
        axis.invert_yaxis()
        #axis.set_xlabel("attributions")
        axis.set_ylabel(cluster_name)

        #plt.show()
    plt.xlabel('attributions')
    plt.savefig(os.path.join(output_dir, f'{prefix}_{plot_name}_attributions.png'))
    plt.close()


def determine_prediction_cluster(error, correct_treshold, incorrect_treshold):
    if error <  (-1) * incorrect_treshold:
        print("overestimated!")
        return 'overestimated'
    elif error > incorrect_treshold:
        return 'underestimated'
    elif (-1) * correct_treshold <= error <= correct_treshold:
        return 'correct'
    else: 
        return None


