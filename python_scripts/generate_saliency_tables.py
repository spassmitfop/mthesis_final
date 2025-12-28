#script for generating the tables showing the object focus comparison results
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
#folder in which the object focus results are saved
results_folder = os.path.join(Path(__file__).parent.parent, "verify_saliency3")
#should only the first seed be used to generate the tables?
use_only_first_seed = False
#which wrapper should appear in the tables?
wrappers_for_table = ["parallelplanes", "obs_mode_dqn"] #["masked_dqn_bin", "masked_dqn_planes", "parallelplanes", "obs_mode_dqn"]
#For which saliency methods should tables be generated?
saliency_methods_for_table = ["Saliency", "GuidedBackprop"]
#For which aggregations methods should tables be generated?
aggregation_methods_for_table = ["mean"] #["mean", "max"]
#Which metrics should appear in the tables and what should they be called in these tables?
metrics_for_table = {"corr_mask_value": "Value (Pearson)",
                        "corr_mask_policy": "Action (Pearson)",
                        "corr_mask_value2": "Value (Spearman)",
                        "corr_mask_policy2": "Action (Spearman)",
                     }
#What should the wrappers be called in the tables?
wrapper_name_to_latex_name = {
        "masked_dqn_bin": "Binary Masks",
        "masked_dqn_planes": "Planes",
        "parallelplanes": "Parallel Planes",
        "obs_mode_dqn": "DQN-like input",
    }

for ext in ["ext0", "ext10",]:
    folder = os.path.join(results_folder, ext)
    if ext == "ext10" or ext == "ext20":
        use_extended = True
    else:
        use_extended = False

    average_corr_per_wrapper = {}

    scores = {}
    per_wrapper_game_mean = {}
    per_wrapper_mean = {}


    for saliency_method in saliency_methods_for_table:
        for aggregation_method in aggregation_methods_for_table:
            for wrapper_for_table in wrappers_for_table:
                json_file = wrapper_for_table + "_" + saliency_method + "_" + aggregation_method + ".json"
                json_file_path = json_file if folder == "" else os.path.join(folder, json_file)
                with open(json_file_path, "r") as file:
                    data = json.load(file)

                for wrapper, dict_for_game in data.items():
                    if wrapper not in scores:
                        scores[wrapper] = {}
                        per_wrapper_game_mean[wrapper] = {}
                        per_wrapper_mean[wrapper] = {}

                    for game, dicts_for_runs in dict_for_game.items():
                        if game not in scores[wrapper]:
                            scores[wrapper][game] = {}
                            per_wrapper_game_mean[wrapper][game] = {}

                        for run_name, dict_for_run in dicts_for_runs.items():
                            # Collect metrics
                            if run_name == "metrics":
                                continue
                            scores[wrapper][game] = dict_for_run['metrics']
                            for m, vs in scores[wrapper][game].items():
                                if len(vs) != 3:
                                    exit()
                            if use_only_first_seed:
                                per_wrapper_game_mean[wrapper][game] = {
                                    m: vs[0] for m, vs in scores[wrapper][game].items()
                                }
                            else:
                                per_wrapper_game_mean[wrapper][game] = {
                                    m: np.mean(vs) for m, vs in scores[wrapper][game].items()
                                }


                    first_game = list(scores[wrapper].keys())[0]
                    metrics_of_wrapper = list(scores[wrapper][first_game].keys())

                    # Per-wrapper means
                    per_wrapper_mean[wrapper] = {
                        m: np.mean([game_means[m] for game_means in per_wrapper_game_mean[wrapper].values()])
                        for m in metrics_of_wrapper
                    }
            if "obs_mode_dqn" in wrappers_for_table:
                bigger_than_dqn = {}
                for wrapper3, g_dict in per_wrapper_game_mean.items():
                    if wrapper3 not in bigger_than_dqn:
                        bigger_than_dqn[wrapper3] = defaultdict(int)

                    for g, m_dict in g_dict.items():
                        for metric, v in m_dict.items():
                            if metric in metrics_for_table:
                                if v > per_wrapper_game_mean["obs_mode_dqn"][g][metric]:
                                    bigger_than_dqn[wrapper3][metric] += 1

            table = """ 
            \\begin{table}[h!]
            \\centering
            \\begin{tabular}{c| 
            """

            table += "c" * len(wrappers_for_table) + "}"
            table += """
            \\toprule"""
            string_wrappers = "&".join(
                wrapper_name_to_latex_name[wrapper].replace("_", "\\_") for wrapper in wrappers_for_table)
            table += f"""
              & {string_wrappers} \\\\
              """
            table += """\midrule
            """
            for metric, metric_name_latex in metrics_for_table.items():
                values = [per_wrapper_mean[wrapper][metric] for wrapper in wrappers_for_table]
                if "obs_mode_dqn" in wrappers_for_table:
                    values2 = [bigger_than_dqn[wrapper][metric] for wrapper in wrappers_for_table]
                sorted_unique = sorted(set(values), reverse=True)

                # Assign colors based on rank
                formatted_values = []
                for idx, v in enumerate(values):
                    rounded_v = round(v, 3)
                    if v == sorted_unique[0]:
                        # 1st place — dark blue
                        formatted_values.append(f"\\textcolor{{cbBlue}}{{\\textit{{{rounded_v}}}}}")
                    elif len(sorted_unique) > 1 and v == sorted_unique[1]:
                        # 2nd place — orange
                        formatted_values.append(f"\\textcolor{{cbBlack}}{{{rounded_v}}}")
                    elif len(sorted_unique) > 2 and v == sorted_unique[2]:
                        # 3rd place — sky blue
                        formatted_values.append(f"\\textcolor{{cbOrange}}{{{rounded_v}}}")
                    elif len(sorted_unique) > 3 and v == sorted_unique[3]:
                        # 4th place — vermillion/red
                        formatted_values.append(f"\\textcolor{{cbRed}}{{\\underline{{{rounded_v}}}}}")
                    else:
                        formatted_values.append(str(rounded_v))

                string_results = metric_name_latex + " & " + " & ".join(formatted_values)

                table += string_results + "\\\\"
            table += """
            \\bottomrule
            \\end{tabular}
            """
            extended_string = "extended" if use_extended else ""
            table += "\\caption{Correlations between " + saliency_method + " and masks, extended by " + str(
                ext)[3:] + " pixels (" + str(1 if use_only_first_seed else 3) + " seeds). The channels were aggregated with " + aggregation_method + ".}"
            extended_string = ext
            table += """
            \\label{tab:corr_table_""" + saliency_method + "_" + aggregation_method + "_" + extended_string + "}"
            table += """
            \\end{table}
            """
            print(table)
