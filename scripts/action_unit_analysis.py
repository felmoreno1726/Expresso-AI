import os, argparse


from expresso_ai.interpret.au_analysis import generate_participant_AU_summary, generate_overall_prediction_clusters, generate_underestimated_prediction_clusters, generate_overestimated_prediction_clusters, generate_correct_prediction_clusters#, generate_regional_prediction_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parses inputs")
    parser.add_argument('--interpretation_path', type=str, default='./interpretation', help="Default='./intepretation'. Defines the path to interpretation directory.")
    parser.add_argument('--algorithm_name', type=str, default='DeepLift', help="Default='DeepLift'. Defines the algorithm to interpret for")
    args = parser.parse_args()
    #Define Tresholds
    CORRECT_TRESHOLD = 5.0
    INCORRECT_TRESHOLD = 15.5
    #participant summary
    CLUSTERS = {
            "minimal": (0, 13),
            "mild": (14, 19),
            "moderate": (20, 28),
            "severe": (29, 63)
    }
    #participant summary
    #clusters
    au_class_names = ['c', 'r']
    for au_class_name in au_class_names:
        PLOT_NAME = f"au_{au_class_name}_attribution_correlations"
        generate_participant_AU_summary(args.interpretation_path, args.algorithm_name, class_name=au_class_name)
        generate_overall_prediction_clusters(args.interpretation_path, args.algorithm_name, CLUSTERS, PLOT_NAME, class_name=au_class_name)
        generate_underestimated_prediction_clusters(args.interpretation_path, args.algorithm_name, CLUSTERS, INCORRECT_TRESHOLD, PLOT_NAME, class_name=au_class_name)
        generate_overestimated_prediction_clusters(args.interpretation_path, args.algorithm_name, CLUSTERS, INCORRECT_TRESHOLD, PLOT_NAME, class_name=au_class_name)
        generate_correct_prediction_clusters(args.interpretation_path, args.algorithm_name, CLUSTERS, CORRECT_TRESHOLD, PLOT_NAME, class_name=au_class_name)
    #for cluster_of_interest in CLUSTERS.keys():
        #generate_regional_prediction_clusters(args.interpretation_path, args.algorithm_name, cluster_of_interest, CLUSTERS, CORRECT_TRESHOLD, INCORRECT_TRESHOLD, PLOT_NAME)
   

