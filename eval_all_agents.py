import os

num_epsiodes = 10
# seeds = [1, 2, 3, 4, 5, 6]
outfolder = "shared_j"
agents_folder = "agents"
script = os.path.join("python_scripts", "eval.py")
wrappers_to_evaluate = [
    'masked_dqn_planes_combined2', 'masked_dqn_bin', 'masked_dqn_planes',
    "double_first_hlayer", "double_input", "obs_mode_obj", "multiply_player_info3",
    "masked_dqn_pixels", "parallelplanes",
    "bin_scaled_1point2", "planes_scaled_1point2",
    "masked_dqn_bin_scaled_1point5", "masked_dqn_planes_scaled_1point5",
    "pixels_plus_obj_no_bnorm", "bin_plus_obj_no_bnorm",
    "bin_scaled_kr_0point87", "planes_scaled_kr_0point87",
    "use_distances2", "use_distances_angle2", "use_distances_angle_mpi2",
    "use_dis_ang_mpi_noxy2", "use_dir_mpi_noxy2", "use_direction_mpi2", "use_direction2",
    "use_all3", "use_overlap2", 'use_angle2', "use_o_dis_angle", "use_vel",
    "use_dis_angle_c", "use_many", "use_t_dis_ang", "pixels_plus_obj_no_bnorm"]

modifications = {"freeway": ["stop_all_cars_edge", "reverse_car_speed_top", "all_black_cars"],
                 "seaquest": ["disable_enemies", "gravity", "random_color_enemies"],
                 "pong": ["lazy_enemy", "up_drift", "hidden_enemy"],
                 "spaceinvaders": ["relocate_shields_off_by_three", "relocate_shields_right"],
                 "amidar": ["pig_enemies", "paint_roller_player"],
                 "boxing": ["switch_positions", "drunken_boxing", "color_player_red"]
                 }

wrapper_to_cmd_string = {
    "double_first_hlayer": "--encoder_dims 512 512 1024 512",
    "double_input": "--double_input",
    'multiply_player_info3': "--multiply_player_info",
    'obs_mode_obj': "",
    'masked_dqn_planes_combined2': "", 'masked_dqn_bin': "", 'masked_dqn_planes': "",
    "masked_dqn_pixels": "", "parallelplanes": "",
    "bin_scaled_1point2": "--scale_w 1.2 --scale_h 1.2",
    "planes_scaled_1point2": "--scale_w 1.2 --scale_h 1.2",
    "masked_dqn_bin_scaled_1point5": "--scale_w 1.5 --scale_h 1.5",
    "masked_dqn_planes_scaled_1point5": "--scale_w 1.5 --scale_h 1.5",
    "bin_scaled_kr_0point87": "--keep_ratio --scale_w 0.87 --scale_h 0.87",
    "planes_scaled_kr_0point87": "--keep_ratio --scale_w 0.87 --scale_h 0.87",
    "pixels_plus_obj_no_bnorm": "",
    "bin_plus_obj_no_bnorm": "",

    "use_distances2": "--use_distances",
    "use_angle2": "--use_angle",
    "use_distances_angle2": "--use_distances --use_angle",
    "use_distances_angle_mpi2": "--use_distances --use_angle --multiply_player_info",
    "use_dir_mpi_noxy2": "--use_direction --multiply_player_info --no-use_object_xy",
    "use_direction2": "--use_direction",
    "use_direction_mpi2": "--use_direction --multiply_player_info",
    "use_dis_ang_mpi_noxy2": "--use_distances --use_angle --multiply_player_info --no-use_object_xy",
    "use_overlap2": "--use_overlap",
    "use_vel": "--use_vel",
    "use_o_dis_angle": "--use_origin_angle --use_origin_distances",
    "use_t_dis_ang": "--use_time_distances --use_time_angle",
    "use_dis_angle_c": "--use_distances --use_angle --apply_centerpoints",
    "use_all3": "--use_distances --use_angle --multiply_player_info --use_direction --use_vel --use_overlap",
    "use_many": "--use_distances --use_angle --multiply_player_info --use_direction --use_overlap",
}
extended_obj = "-mw ext_obj"
mw = {
    "use_vel": extended_obj,
    "use_o_dis_angle": extended_obj,
    "bin_scaled_kr_0point87": "-mw masked_dqn_bin_scaled",
    "planes_scaled_kr_0point87": "-mw masked_dqn_planes_scaled",
    "masked_dqn_bin_scaled_1point5": "-mw masked_dqn_bin_scaled",
    "bin_scaled_1point2": "-mw masked_dqn_bin_scaled",
    "pixels_plus_obj_no_bnorm": "-mw masked_dqn_pixels_plus_og_obj",
    "use_t_dis_ang": extended_obj,
    "bin_plus_obj_no_bnorm": "-mw masked_dqn_bin_plus_og_obj",
    "use_dis_angle_c": extended_obj,
    "multiply_player_info3": extended_obj,
    "planes_scaled_1point2": "-mw masked_dqn_planes_scaled",
    "use_all3": extended_obj,
    "use_many": extended_obj,
    "masked_dqn_planes_scaled_1point5": "-mw masked_dqn_planes_scaled",
    'masked_dqn_planes_combined2': "-mw masked_dqn_planes_combined2", 'masked_dqn_bin': "-mw masked_dqn_bin",
    'masked_dqn_planes': "-mw masked_dqn_planes",
    "masked_dqn_pixels": "-mw masked_dqn_pixels", "parallelplanes": "-mw masked_dqn_parallelplanes",
    'double_first_hlayer': extended_obj,
    'double_input': extended_obj,
    'obs_mode_obj': extended_obj,
    'use_distances2': extended_obj,
    'use_distances_angle2': extended_obj,
    'use_distances_angle_mpi2': extended_obj,
    'use_dis_ang_mpi_noxy2': extended_obj,
    'use_dir_mpi_noxy2': extended_obj,
    'use_direction_mpi2': extended_obj,
    'use_direction2': extended_obj,
    'use_overlap2': extended_obj,
    'use_angle2': extended_obj,
}

archi = {
    "use_vel": "PPO_OBJ",
    "use_o_dis_angle": "PPO_OBJ",
    "bin_scaled_kr_0point87": "PPO",
    "planes_scaled_kr_0point87": "PPO",
    "masked_dqn_bin_scaled_1point5": "PPO",
    "bin_scaled_1point2": "PPO",
    "pixels_plus_obj_no_bnorm": "PPOCombi2Big",
    "use_t_dis_ang": "PPO_OBJ",
    "bin_plus_obj_no_bnorm": "PPOCombi2Big",
    "use_dis_angle_c": "PPO_OBJ",
    "multiply_player_info3": "PPO_OBJ",
    "planes_scaled_1point2": "PPO",
    "use_all3": "PPO_OBJ",
    "use_many": "PPO_OBJ",
    "masked_dqn_planes_scaled_1point5": "PPO",
    'masked_dqn_planes_combined2': "PPO",
    'masked_dqn_bin': "PPO",
    'masked_dqn_planes': "PPO",
    "masked_dqn_pixels": "PPO",
    "parallelplanes": "PPO",
    'double_first_hlayer': "PPO_OBJ",
    'double_input': "PPO_OBJ",
    'obs_mode_obj': "PPO_OBJ",
    'use_distances2': "PPO_OBJ",
    'use_distances_angle2': "PPO_OBJ",
    'use_distances_angle_mpi2': "PPO_OBJ",
    'use_dis_ang_mpi_noxy2': "PPO_OBJ",
    'use_dir_mpi_noxy2': "PPO_OBJ",
    'use_direction_mpi2': "PPO_OBJ",
    'use_direction2': "PPO_OBJ",
    'use_overlap2': "PPO_OBJ",
    'use_angle2': "PPO_OBJ",
}
game_to_env_id = {"boxing": "ALE/Boxing-v5", "pong": "ALE/Pong-v5", "spaceinvaders": "ALE/SpaceInvaders-v5",
                  "freeway": "ALE/Freeway-v5", "amidar": "ALE/Amidar-v5", "seaquest": "ALE/Seaquest-v5", }

from collections import Counter

list1 = list(archi.keys())  # new_agents
list2 = list(mw.keys())
c1 = Counter(list1)
c2 = Counter(list2)

missing_in_list2 = c1 - c2
missing_in_list1 = c2 - c1
print("Missing in list2:", list(missing_in_list2.elements()))
print("Missing in list1:", list(missing_in_list1.elements()))

not_existing_agents = []
for agent in wrappers_to_evaluate:
    seeds = [1, 2, 3, 4, 5, 6] if "PPO_OBJ" == archi[agent] else [1, 2, 3]
    for seed in seeds:
        for game in modifications.keys():
            agent1 = os.path.join(agents_folder, game, str(seed), agent + ".cleanrl_model")
            if not os.path.exists(agent1):
                not_existing_agents.append((agent, game, seed))
if len(not_existing_agents) > 0:
    print(not_existing_agents)
    exit(1)

for agent in wrappers_to_evaluate:
    seeds = [1, 2, 3, 4, 5, 6] if "PPO_OBJ" == archi[agent] else [1, 2, 3]
    print(agent, seeds)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    for game, modifications_list in modifications.items():
        print(game)
        list_agents = [os.path.join(agents_folder, game, str(seed), agent + ".cleanrl_model") for seed in seeds]

        agents_string = " ".join(list_agents)
        modif = ""
        out_file = game + ".json"
        out = os.path.join(outfolder, out_file)
        if not os.path.exists(out):
            print(f"{out_file} does not exist, will be created")
        cmd = f"python {script} -g {game_to_env_id[game]} -a {agents_string} -ar {archi[agent]} -out {out} {modif} -e {num_epsiodes} -n {agent} -ap {mw[agent]} -obs obj " + \
              wrapper_to_cmd_string[agent]
        os.system(cmd)
        for m in modifications_list:
            print(m)
            modif = "-m " + m
            out_file = m + ".json"
            out = os.path.join(outfolder, out_file)
            if not os.path.exists(out):
                print(f"{out_file} does not exist, will be created")

            cmd = f"python {script} -g {game_to_env_id[game]} -a {agents_string} -ar {archi[agent]} -out {out} {modif} -e {num_epsiodes} -n {agent} -ap {mw[agent]} -obs obj " + \
                  wrapper_to_cmd_string[agent]
            os.system(cmd)
