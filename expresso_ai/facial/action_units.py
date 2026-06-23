import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..utils.video_helpers import get_frame_indices
from ..interpret.cluster import generate_cluster_plot


def load_participant_AU(au_data_path, participant_id):
    """
    Loads Action Units (AU) for a given participant given its sample position, window_size and input_stride used.
    """
    au_path = os.path.join(au_data_path, str(participant_id), 'aus.csv')
    print("au path: ", au_path)
    au_dataframe = pd.read_csv(au_path)
    return au_dataframe


def get_sample_AUS(au_dataframe, is_valid_frames, window_size, window_index, overlap, input_dilation):
    #filter au_dataframe for the valid_frames
    au_dataframe = au_dataframe.iloc[np.where(is_valid_frames)]
    #get sample frames
    frame_indices = get_frame_indices(window_size, window_index, overlap, input_dilation)
    #get sample aus
    au_dataframe = au_dataframe.iloc[frame_indices]
    return au_dataframe


def get_effective_AUS(au_dataframe):
    """
    Returns a dataframe that corresponds to the input dataframe, but contains only columns corresponding to ActionUnits
    au_dataframe (pd.DataFrame): an input dataframe containing Action Units
    returns au_dataframe filtered by only columns containing Action Unit scores (AUs) of the form "AU#_c"
    """
    return au_dataframe.filter(regex="AU")


def compute_au_correlations(au, framewise_attributions):
    """
    computes the cross-correlation between columns of 
    """
    au['framewise_attributions'] = framewise_attributions
    return au.corr()


