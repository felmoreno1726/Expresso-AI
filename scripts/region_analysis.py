import os, argparse

from expresso_ai.interpret.ModelInterpreter import ModelInterpreter
from expresso_ai.interpret.regionwise_analysis import generate_participant_region_summary, generate_prediction_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parses inputs")
    parser.add_argument('--interpretation_path', type=str, default='./interpretation', help="Default='./intepretation'. Defines the path to interpretation directory.")
    parser.add_argument('--algorithm_name', type=str, default='DeepLift', help="Default='DeepLift'. Defines the algorithm to interpret for")
    args = parser.parse_args()
    CORRECT_TRESHOLD = 5.0
    INCORRECT_TRESHOLD = 15.5
    #participant summary
    CLUSTERS = {
            "minimal": (0, 13),
            "mild": (14, 19),
            "moderate": (20, 28),
            "severe": (29, 63)
    }
    #clusters
    #for data_type in ['raw']:
    for data_type in ['raw', 'normalized']:
        #generate_participant_region_summary(args.interpretation_path, args.algorithm_name, data_type)
        for decision_type in ['overall', 'correct', 'overestimated', 'underestimated']:
            #continue
            generate_prediction_clusters(args.interpretation_path, args.algorithm_name, CLUSTERS, CORRECT_TRESHOLD, INCORRECT_TRESHOLD, decision_type, data_type)
    #for cluster_of_interest in ModelInterpreter.CLUSTERS.keys():
        #generate_regional_prediction_clusters(args.interpretation_path, args.algorithm_name, cluster_of_interest, CLUSTERS, CORRECT_TRESHOLD, INCORRECT_TRESHOLD)
