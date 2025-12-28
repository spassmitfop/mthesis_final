import json
import os
from pathlib import Path

import numpy as np
import rliable.metrics as rlm
from collections import defaultdict

from reference_scores.scores_for_hns import atari_scores

use_all_seeds = True #should all seeds be used to generate the plots and tables?
use_hns = False #should human normalized scores or maximum normalized scores be used?
store_min_max_dict = False #should the minimum and maximum scores for each environment be saved in a json file?
random_jsons_folder =  os.path.join(Path(__file__).parent.parent, "random_jsons") #folder in which the scores of the random agent are saved
performance_results_folder = os.path.join(Path(__file__).parent.parent, "performance_results") #folder in which the performance results are saved
max_dict_json_folder = os.path.join(Path(__file__).parent.parent, "reference_scores") #folder in which the minimum and maximum scores for each environement are saved.
#These mini
number_of_seeds_to_use = 3 # only relevant if !use_all_seeds

modifications = {"pong": ["lazy_enemy", "up_drift", "hidden_enemy"],
                 "spaceinvaders": ["relocate_shields_off_by_three", "relocate_shields_right"],
                 "freeway": ["stop_all_cars_edge", "reverse_car_speed_top", "all_black_cars"],
                 "amidar": ["pig_enemies", "paint_roller_player"],
                 "seaquest": ["gravity", "random_color_enemies"],  # "disable_enemies"
                 "boxing": ["switch_positions", "drunken_boxing", "color_player_red"]
                 }
base_games = list(modifications.keys())
# base_games = []
modifs = [item for sublist in list(modifications.values()) for item in sublist]
visual_modifs = [("all_black_cars", "freeway"), ("random_color_enemies", "seaquest"), ("color_player_red", "boxing"), ]
logic_modifs = [("lazy_enemy", "pong"), ("up_drift", "pong"), ("hidden_enemy", "pong"),
                ("relocate_shields_off_by_three", "spaceinvaders"), ("relocate_shields_right", "spaceinvaders"),
                ("stop_all_cars_edge", "freeway"), ("reverse_car_speed_top", "freeway"), ("gravity", "seaquest"),
                ("switch_positions", "boxing"), ("drunken_boxing", "boxing")]

relevant_wrappers = ["double_first_hlayer", "double_input", "obs_mode obj",
                     'masked_dqn_planes_combined2', 'masked_dqn_bin', 'masked_dqn_planes',
                     "masked_dqn_pixels", "parallelplanes",
                     "use_distances2", "use_distances_angle2", "use_distances_angle_mpi2",
                     "use_dis_ang_mpi_noxy2", "use_dir_mpi_noxy2", "use_direction_mpi2", "use_direction2",
                     "use_all3", "use_overlap2", 'use_angle2', "use_o_dis_angle", "use_vel",
                     "pixels_plus_obj_no_bnorm", "bin_plus_obj_no_bnorm", "bin_plus_obj_no_bnorm2",
                     "bin_scaled_kr_0point87", "planes_scaled_kr_0point87",
                     "bin_scaled_1point2", "planes_scaled_1point2",
                     "use_dis_angle_c", "multiply_player_info3", "use_many",
                     'masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5',
                     'masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5',
                     "masked_dqn_bin_scaled_1point5", "masked_dqn_planes_scaled_1point5",
                     "obs_mode_obj","obs_mode_obj2",
                     ]

#relevant_wrappers = ["parallelplanes_bugged", "use_t_dis_ang", "obs_mode obj", "parallelplanes"]

seeds_per_relevant_wrappers = {
    w: [i for i in range(number_of_seeds_to_use)] for w in relevant_wrappers
}

wrapper_to_name_latex = {'masked_dqn_planes': "Planes",
                         'masked_dqn_planes_combined2': "Planes MPI",
                         'masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5': "Planes Scaled 2/3",
                         'masked_dqn_bin': "Binary Masks",
                         'masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5': "Binary Masks Scaled 2/3",
                         'masked_dqn_pixels': "Object Masks",
                         'double_first_hlayer': "Double First Hlayer",
                         'obs_mode_obj': "Original Semantic Vector",
                         "parallelplanes": "Parallel Plane Masks",
                         "parallelplanes_bugged": "Bugged Parallel Planes Masks",
                         "use_distances2": "Distances",
                         "use_distances_angle2": "Distances, Angles",
                         "use_distances_angle_mpi2": "Distances, Angles, MPI",
                         "use_dis_ang_mpi_noxy2": "Distances, Angles, MPI (noXY)",
                         "use_angle2": "Angles",
                         "use_dir_mpi_noxy2": "Directions, MPI (noXY)",
                         "use_direction_mpi2": "Directions, MPI",
                         "use_direction2": "Directions",
                         "use_overlap2": "Overlaps",
                         "use_vel": "Velocities",
                         "use_o_dis_angle": "Origin Distances, Angles",
                         "bin_scaled_kr_0point87": "Binary Mask Original Ratio",
                         "planes_scaled_kr_0point87": "Planes Original Ratio",
                         "bin_scaled_0point8": "Binary Masks 5/4",
                         "bin_scaled_1point2": "Binary Masks Scaled 5/6",
                         "pixels_plus_obj_no_bnorm": "Object Masks + SV",
                         "use_t_dis_ang": "Time Distances, Angles",
                         "bin_plus_obj_no_bnorm": "Binary Masks + SV",
                         "use_dis_angle_c": "Distances, Angles (Centerpoints)",
                         "multiply_player_info3": "Multiply Player Info",
                         "bin_plus_obj_no_bnorm2": "Binary Masks + SV",
                         "use_all3": "All Extensions of SV Combined",
                         "planes_scaled_1point2": "Planes Scaled 5/6",
                         "use_many": "Many Extensions of SV Combined",
                         "masked_dqn_bin_scaled_1point5": "Binary Masks Scaled 2/3",
                         "masked_dqn_planes_scaled_1point5": "Planes Scaled 2/3",
                         "obs_mode obj": "Original Semantic Vector",
                         'double_input': "Double Input",
                         }


def highlight_value(v):
    return f"\\textcolor{{red}}{{\\textbf{{{v}}}}}"


files_base_game = [
    (g + ".json", g)
    for g in base_games
]

files_modifs = [
    (g + ".json", g)
    for g in modifs
]

files_random_modifs = [
    (g + "_random.json", g)
    for g in modifs
]

files_visual_modifs = [
    (g + ".json", g)
    for (g, b) in visual_modifs
]

files_logic_modifs = [
    (g + ".json", g)
    for (g, b) in logic_modifs
]

files = files_modifs + files_base_game
# files = files_base_game
# files = files_modifs

files_random = [
    (g + "_random.json", g)
    for g in base_games + modifs
]

random_scores = {}

scores_raw_all_games = {}
scores_iqm_per_seed_all_games = {}
scores_iqm_over_seeds_all_games = {}
scores_std_over_seeds_all_games = {}

wrappers_found_in_json = set()


def extract_scores(json_file):
    json_file_path = json_file if performance_results_folder == "" else os.path.join(performance_results_folder, json_file)
    with open(json_file_path, "r") as file:
        data = json.load(file)  # Parse JSON into a Python dictionary
    scores_raw = defaultdict(list)
    scores_iqm_over_seeds = {}
    scores = defaultdict(list)
    scores_std_over_seeds = {}
    found_wrapper_names = []
    for eval_run in data:
        values = list(eval_run.values())
        name = values[0]["Args"]["name"]
        if name in relevant_wrappers:
            if name not in found_wrapper_names:
                found_wrapper_names.append(name)
                wrappers_found_in_json.add(name)
            game = values[0]["Args"]["game"]
            score_list = []
            score_list_iqms = []
            for results in values:
                score_list.append(results["episode_rewards"])
                score_list_iqms.append(results["iqm_reward"])
                # score_list_iqms.append(np.mean(results["episode_rewards"]))

            scores_raw[name] += score_list
            scores[name] += score_list_iqms

    for n in found_wrapper_names:
        if use_all_seeds:
            scores_iqm_over_seeds[n] = rlm.aggregate_iqm(np.array(scores_raw[n]).flatten())
            scores_std_over_seeds[n] = np.std(np.array(scores_raw[n]).flatten())
        else:
            indices = seeds_per_relevant_wrappers[n]
            scores_iqm_over_seeds[n] = rlm.aggregate_iqm(np.array(scores_raw[n])[indices].flatten())
            scores_std_over_seeds[n] = np.std(np.array(scores_raw[n])[indices].flatten())
    return scores_raw, scores, scores_iqm_over_seeds, scores_std_over_seeds


def extract_random_scores(json_file):
    json_file_path = json_file if random_jsons_folder == "" else os.path.join(random_jsons_folder, json_file)
    with open(json_file_path, "r") as file:
        data = json.load(file)  # Parse JSON into a Python dictionary
    if len(data) != 1:
        raise RuntimeError("Something wrong in json structure of random agent")
    eval_run = data[0]
    values = list(eval_run.values())
    if len(values) != 1:
        raise RuntimeError("Something wrong in json structure of random agent")
    results = values[0]
    if len(results["Args"]["modifs"]) == 0 and json_file in files_random_modifs:
        raise RuntimeError(f"No modifs found in json file: {json_file_path}")
    score_list_episodes_rewards = results["episode_rewards"]
    score_list_iqm = results["iqm_reward"]

    return score_list_iqm, score_list_episodes_rewards


for json_file, game in files_random:
    random_iqm_s, random_raw_s = extract_random_scores(json_file)
    random_scores[game] = random_iqm_s

for json_file, game in files:
    s_raw, s, s_iqm_over_seeds, s_std_over_seeds = extract_scores(json_file)
    scores_raw_all_games[game] = s_raw
    scores_iqm_per_seed_all_games[game] = s
    scores_iqm_over_seeds_all_games[game] = s_iqm_over_seeds
    scores_std_over_seeds_all_games[game] = s_std_over_seeds

# remove alternative versions of wrappers, so only one version is included in the plots and tables
if "bin_plus_obj_no_bnorm2" in wrappers_found_in_json and "bin_plus_obj_no_bnorm" in relevant_wrappers:
    relevant_wrappers.remove("bin_plus_obj_no_bnorm")
else:
    if "bin_plus_obj_no_bnorm" in wrappers_found_in_json and "bin_plus_obj_no_bnorm2" in relevant_wrappers:
        relevant_wrappers.remove("bin_plus_obj_no_bnorm2")

if "obs_mode obj" in wrappers_found_in_json and "obs_mode_obj" in relevant_wrappers:
    relevant_wrappers.remove("obs_mode_obj")
else:
    if "obs_mode_obj" in wrappers_found_in_json and "obs_mode obj" in relevant_wrappers:
        relevant_wrappers.remove("obs_mode obj")

if 'masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5' in wrappers_found_in_json and 'masked_dqn_bin_scaled_1point5' in relevant_wrappers:
    relevant_wrappers.remove("masked_dqn_bin_scaled_1point5")
else:
    if 'masked_dqn_bin_scaled_1point5' in wrappers_found_in_json and 'masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5' in relevant_wrappers:
        relevant_wrappers.remove('masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5')

if 'masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5' in wrappers_found_in_json and 'masked_dqn_planes_scaled_1point5' in relevant_wrappers:
    relevant_wrappers.remove("masked_dqn_planes_scaled_1point5")
else:
    if 'masked_dqn_planes_scaled_1point5' in wrappers_found_in_json and 'masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5' in relevant_wrappers:
        relevant_wrappers.remove('masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5')


def all_same_keys(dicts):
    l = list(dicts.items())
    first_keys = set(l[0][1].keys())
    first_game = l[0][0]
    for game, d in l:
        if first_keys != set(d.keys()):
            print(first_keys)
            print(set(d.keys()))
            raise RuntimeError(first_game + " and " + game + " don't have the same keys")


all_same_keys(scores_raw_all_games)


def get_min_max(s):
    min_dict = {game: min(scores.values()) for game, scores in s.items()}
    max_dict = {game: max(scores.values()) for game, scores in s.items()}
    return min_dict, max_dict


count_greater_0 = 0
count_smaller_0 = 0


def compute_normalized_means(s_d, min_d, max_d):
    global count_greater_0, count_smaller_0

    n = {}
    for key, value in s_d.items():
        if len(n) == 0:
            n = {key2: [] for key2 in value.keys()}
        for key2, value2 in value.items():
            if min_d[key] == max_d[key]:
                n[key2].append(0)
            else:
                if max_d[key] - min_d[key] < 0:
                    print(key, "???????????????????")
                if value2 - min_d[key] < 0:
                    count_smaller_0 += 1
                else:
                    count_greater_0 += 1
                n[key2].append((value2 - min_d[key]) / (max_d[key] - min_d[key]))
    return {k: np.mean(v) for k, v in n.items()}


def compute_normalized_means_for_keys(s_d, min_d, max_d, keys):
    n = {}
    for key in keys:
        value = s_d[key]
        if len(n) == 0:
            n = {key2: [] for key2 in value.keys()}
        for key2, value2 in value.items():
            if min_d[key] == max_d[key]:
                n[key2].append(0)
            else:
                n[key2].append((value2 - min_d[key]) / (max_d[key] - min_d[key]))
    return {k: np.mean(v) for k, v in n.items()}


def sort_dict(d, reverse=False):
    return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=reverse))


def compute_normalized_for_raw_scores(s_d, min_d, max_d):
    n = {}
    for game1, wrapper_score_d in s_d.items():
        n[game1] = {}
        for wrapper1, scores in wrapper_score_d.items():
            scores_array = np.array(scores)
            if min_d[game1] == max_d[game1]:
                n[game1][wrapper1] = np.zeros_like(scores_array)
            else:
                n[game1][wrapper1] = (scores_array - min_d[game1]) / (max_d[game1] - min_d[game1])
    return n


def compute_mean_normalized_scores_per_seed(s_d, min_d, max_d):
    normalized_scores = compute_normalized_for_raw_scores(s_d, min_d, max_d)
    r = {}
    if use_all_seeds:
        for wrapper1 in relevant_wrappers:
            # Get all lists of scores for this wrapper across all games
            all_scores = [game_dict[wrapper1] for game_dict in normalized_scores.values()]

            # Transpose the list of lists to group by index and compute mean
            r[wrapper1] = [np.mean(scores_at_j) for scores_at_j in zip(*all_scores)]
    else:
        for wrapper1, seeds1 in seeds_per_relevant_wrappers.items():
            r[wrapper1] = []
            for seed in seeds1:
                l = []
                for game1, game_dict in normalized_scores.items():
                    l.append(game_dict[wrapper1][seed])
                r[wrapper1].append(np.mean(l))
    return r


def print_dict(d):
    for key, value in d.items():
        if isinstance(value, list):
            value2 = [round(x, 3) for x in value]
        else:
            value2 = round(value, 3)
        print(f"{key}: {value2}")


if use_hns:
    max_dict = {}
    min_dict = {}
    for game1, modifications1 in modifications.items():
        max_dict[game1] = atari_scores[game1][1]
        min_dict[game1] = atari_scores[game1][0]
        for modification1 in modifications1:
            max_dict[modification1] = atari_scores[game1][1]
            min_dict[modification1] = atari_scores[game1][0]
else:
    min_dict, max_dict = get_min_max(scores_iqm_over_seeds_all_games)
    if store_min_max_dict:
        with open(os.path.join(max_dict_json_folder, "min_dict.json"), "w") as f:
            json.dump(min_dict, f, indent=4)  # indent=4 makes it pretty-printed
        with open(os.path.join(max_dict_json_folder, "max_dict.json"), "w") as f:
            json.dump(max_dict, f, indent=4)  # indent=4 makes it pretty-printed

    with open(os.path.join(max_dict_json_folder, "min_dict.json"), "r") as f:
        min_dict = json.load(f)

    with open(os.path.join(max_dict_json_folder, "max_dict.json"), "r") as f:
        max_dict = json.load(f)

normalized_scores_per_seed = compute_mean_normalized_scores_per_seed(scores_iqm_per_seed_all_games, random_scores,
                                                                     max_dict)
normalized_scores_with_random = sort_dict(
    compute_normalized_means(scores_iqm_over_seeds_all_games, random_scores,
                             max_dict), reverse=True)

normalized_scores_with_min = sort_dict(
    compute_normalized_means(scores_iqm_over_seeds_all_games, min_dict,
                             max_dict), reverse=True)

normalized_scores_with_random_basegames = sort_dict(
    compute_normalized_means_for_keys(scores_iqm_over_seeds_all_games, random_scores,
                                      max_dict, base_games), reverse=True)
visual_modifs_list = [m for (m, b) in visual_modifs]
logic_modifs_list = [m for (m, b) in logic_modifs]
normalized_scores_with_random_vmods = sort_dict(
    compute_normalized_means_for_keys(scores_iqm_over_seeds_all_games, random_scores,
                                      max_dict, visual_modifs_list), reverse=True)
normalized_scores_with_random_lmods = sort_dict(
    compute_normalized_means_for_keys(scores_iqm_over_seeds_all_games, random_scores,
                                      max_dict, logic_modifs_list), reverse=True)
normalized_scores_with_random_mods = sort_dict(
    compute_normalized_means_for_keys(scores_iqm_over_seeds_all_games, random_scores,
                                      max_dict, modifs), reverse=True)
obs_mode_obj = "obs_mode obj" if "obs_mode obj" in relevant_wrappers else (
    "obs_mode_obj" if "obs_mode_obj" in relevant_wrappers else None)

baselines = {
             'masked_dqn_planes': ['masked_dqn_planes_combined2', "parallelplanes",
                                   "parallelplanes_bugged",
                                   "planes_scaled_1point2",
                                   'masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5'
                                   "planes_scaled_kr_0point87", ],
             'masked_dqn_bin': ["bin_plus_obj_no_bnorm", "bin_plus_obj_no_bnorm2",
                                'masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5', "bin_scaled_1point2",
                                "bin_scaled_kr_0point87", ],
             "masked_dqn_pixels": ['masked_dqn_bin_obj --scale 1.2', 'masked_dqn_bin_obj --scale 1.5',
                                   "pixels_plus_obj_no_bnorm", ],
             "parallelplanes": ["parallelplanes_bugged"]
             }
if obs_mode_obj is not None:
    baselines[obs_mode_obj] = ["multiply_player_info3",
                                  "double_first_hlayer", "double_input",
                                  "use_distances2", "use_angle2",
                                  "use_distances_angle2", "use_distances_angle_mpi2", "use_dis_ang_mpi_noxy2",
                                  "use_o_dis_angle", "use_dis_angle_c",
                                  "use_direction2", "use_direction_mpi2", "use_dir_mpi_noxy2",
                                  "use_overlap2", "use_all2", "use_vel",
                                  "pixels_plus_obj_no_bnorm", "bin_plus_obj_no_bnorm2"
        , "use_t_dis_ang", "use_many", "use_all3", ]

games = [x[1] for x in files]