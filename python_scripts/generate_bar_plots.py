# generates bar plots visualizing the agents' performances
from extract_scores_from_json import *
# In extract_scores_from_json, you can specify which jsons to use for generating the bar plots
import matplotlib.pyplot as plt

display_std_in_bar_plots = False

# specify which baselines and which bars to include in each plot
baselines_bars = {(obs_mode_obj, "masked_dqn_bin", "masked_dqn_pixels"):
    [
        "use_dis_ang_mpi_noxy2",
        "use_dir_mpi_noxy2",
        "use_overlap2", "use_vel",
        "bin_plus_obj_no_bnorm2", "pixels_plus_obj_no_bnorm",
        "use_all3",
    ],
    ('masked_dqn_planes',): ['masked_dqn_planes_combined2', 'masked_dqn_planes_combined',
                             'masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5',
                             "parallelplanes",
                             "planes_scaled_kr_0point87", "planes_scaled_1point2",
                             "planes_scaled_0point8",
                             ],
    ('masked_dqn_bin',): ['masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5',
                          "bin_scaled_kr_0point87", "bin_scaled_0point8", "bin_scaled_1point2",
                          "bin_plus_obj_no_bnorm", "bin_plus_obj_no_bnorm2",
                          ],
}
# the names used in the plots
bar_plots_names_dict = {
    "obs_mode obj": "Original SV",
    "obs_mode_obj": "Original SV",
    "use_dis_ang_mpi_noxy2": "DisAng",
    "use_dir_mpi_noxy2": "Dir",
    "use_overlap2": "Overlaps", "use_vel": "Velocities",
    "pixels_plus_obj_no_bnorm": "SV+OM",
    "use_all3": "All", "use_many": "Many",
    "bin_plus_obj_no_bnorm2": "SV+BM",
    "bin_plus_obj_no_bnorm": "SV+BM",
    'masked_dqn_planes': "Plane M.", 'masked_dqn_planes_combined2': "Planes MPI",
    'masked_dqn_planes_scaled --scale_h 1.5 --scale_w 1.5': "Scaled 2/3",
    "parallelplanes": "Parallel",
    "planes_scaled_kr_0point87": "Original AR", "planes_scaled_1point2": "Scaled 5/6",
    'masked_dqn_bin': "Binary M.",
    'masked_dqn_bin_scaled --scale_h 1.5 --scale_w 1.5': "Scaled 2/3",
    "bin_scaled_kr_0point87": "Original AR", "bin_scaled_1point2": "Scaled 5/6",
    "masked_dqn_pixels": "Object M.",
    "masked_dqn_planes_scaled_1point5": "Scaled 2/3",
    "masked_dqn_bin_scaled_1point5": "Scaled 2/3",
}

for idx_o, (baselines_l, wrappers) in enumerate(baselines_bars.items()):
    categories = []
    values_l = []
    for wrapper in wrappers:
        if wrapper in relevant_wrappers:
            categories.append(bar_plots_names_dict[wrapper])
            values_l.append(normalized_scores_with_random[wrapper])

    values = np.array(values_l)
    if idx_o == 0:
        plt.figure(figsize=(8, 4))
    else:
        plt.figure(figsize=(6, 4))

    stds_over_seeds = [np.std(normalized_scores_per_seed[wrapper1]) for wrapper1 in wrappers if
                       wrapper1 in relevant_wrappers]
    if display_std_in_bar_plots:
        plt.bar(categories, values, color='skyblue', yerr=stds_over_seeds, capsize=5)
    else:
        plt.bar(categories, values, color='skyblue', capsize=5)

    colors = ["#D55E00", "#F0E442", "#CC79A7", "#E69F00", "#009E73"]
    for idx, b in enumerate(baselines_l):
        if b in relevant_wrappers:
            baseline_value = normalized_scores_with_random[b]
            plt.axhline(y=baseline_value, color=colors[idx], linestyle='--',
                        label=f'{bar_plots_names_dict[b]}')

    plt.xlabel('Input Representation')
    plt.ylabel('Mean MNS')
    # Move legend outside the plot
    if idx_o == 0:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend()

    plt.tight_layout()
    if baselines_l[0] is not None:
        plt.savefig(baselines_l[0] + ".png")
    plt.show()
