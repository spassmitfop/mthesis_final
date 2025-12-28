# Script for evaluating agents.
from hackatari import HackAtari, HumanPlayable
import numpy as np
import sys
import torch
import os
import argparse
import json
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
import rliable.metrics as rlm
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ocatari_wrappers
from reference_scores.scores_for_hns import atari_scores

# Disable graphics window (SDL) for headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"


class HackAtariArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        # Check if `-h` or `--help` is in the arguments
        if args is None:
            args = sys.argv[1:]
        if '-h' in args or '--help' in args:
            if not '-g' in args or '--game' in args:
                print(
                    "Call the script with a given game to get a list of available modifications.")
            else:
                print(hackatari._available_modifications(
                    args[args.index('-g') + 1]))
                print(
                    "\n provide -h (or --help) without a game argument for the original help message.")
                exit(0)

        # Call the original `parse_args` method to display the default help
        return super().parse_args(args, namespace)


def calculate_hns(score, game):
    human_score = atari_scores[game.lower()][1]
    random_score = atari_scores[game.lower()][0]
    return (score - random_score) / (human_score - random_score)


def main():
    parser = HackAtariArgumentParser(description="HackAtari Experiment Runner")
    parser.add_argument("-g", "--game", type=str,
                        default="Seaquest", help="Game to be run")
    parser.add_argument("-obs", "--obs_mode", type=str,
                        default="dqn", help="Observation mode (ori, dqn, obj)")
    parser.add_argument("-w", "--window", type=int,
                        default=4, help="Buffer window size")
    parser.add_argument("-f", "--frameskip", type=int,
                        default=4, help="Frames skipped after each action")
    parser.add_argument("-dp", "--dopamine_pooling", action='store_true',
                        help="Enable dopamine-like frameskipping")
    parser.add_argument("-m", "--modifs", nargs="+",
                        default=[], help="List of modifications to apply")
    parser.add_argument("-rf", "--reward_function", type=str,
                        default="", help="Custom reward function path")
    parser.add_argument("-a", "--agents", nargs='+',
                        required=True, help="List of trained agent model paths")
    parser.add_argument("-mo", "--game_mode", type=int,
                        default=0, help="Alternative ALE game mode")
    parser.add_argument("-d", "--difficulty", type=int,
                        default=0, help="Alternative ALE difficulty")
    parser.add_argument("-e", "--episodes", type=int,
                        default=10, help="Number of episodes per agent")
    parser.add_argument("-wr", "--wrapper", type=str,
                        default="", help="Use a masking wrapper")
    parser.add_argument("-out", "--output", type=str,
                        default="results.json", help="Output file for results")
    parser.add_argument("-ar", "--architecture", type=str,
                        default="PPO", help="")
    parser.add_argument("-cu", "--cuda", type=bool,
                        default=True, help="")
    parser.add_argument("-mw", "--masked_wrapper", type=str,
                        default="", help="Use a masking wrapper like masked_dqn_bin or the extended obj vector (ext_obj)")
    parser.add_argument("-n", "--name", type=str,
                        default="no name", help="Name of the agents")
    #The following argument sets the respective parameters of the ObjExtended wrapper
    parser.add_argument("-udis", "--use_distances", action='store_true')
    parser.add_argument("-udir", "--use_direction", action='store_true')
    parser.add_argument("-uwh", "--use_wh", action='store_true')
    parser.add_argument("-rs", "--reshape", type=bool, default=True)
    parser.add_argument("-mpi", "--multiply_player_info", action='store_true')
    parser.add_argument("-ua", "--use_angle", action='store_true')
    parser.add_argument("-uo", "--use_overlap", action='store_true')
    parser.add_argument("-uc", "--use_centerpoints", action='store_true')
    parser.add_argument("-apc", "--apply_centerpoints", action='store_true')
    parser.add_argument("-uv", "--use_vel", action='store_true')
    parser.add_argument("-uoa", "--use_origin_angle", action='store_true')
    parser.add_argument("-uod", "--use_origin_distances", action='store_true')
    parser.add_argument("-uta", "--use_time_angle", action='store_true')
    parser.add_argument("-utd", "--use_time_distances", action='store_true')
    parser.add_argument("-ovo", "--overlap_offset", type=int, default=2)
    parser.add_argument("-di", "--double_input", action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_object_xy", dest="use_object_xy", action="store_true", )
    group.add_argument("--no-use_object_xy", dest="use_object_xy", action="store_false", )
    parser.set_defaults(use_object_xy=True)

    parser.add_argument("-sw", "--scale_w", type=float, default=1.0)
    parser.add_argument("-sh", "--scale_h", type=float, default=1.0)
    parser.add_argument("-kr", "--keep_ratio", action='store_true')


    parser.add_argument("-ap", "--agent_paths",action='store_true', help="Paths where the agents to be evaluated are saved")

    parser.add_argument(
        "-ed", "--encoder_dims",
        type=int, nargs="+",
        default=[256, 512, 1024, 512],
        help="Layer dimensions of the MLP agent before nn.Flatten()"
    )
    parser.add_argument(
        "-dd", "--decoder_dims",
        type=int, nargs="+",
        default=[512],
        help="Layer dimensions of the MLP agent before nn.Flatten()"
    )

    args = parser.parse_args()
    s = args.game
    if s.endswith("-v5"):
        s = s[:-3]
    if s.startswith("ALE/"):
        s = s[len("ALE/"):]
    game_name = s
    env = HackAtari(
        args.game, args.modifs, args.reward_function,
        dopamine_pooling=args.dopamine_pooling, game_mode=args.game_mode,
        difficulty=args.difficulty, render_mode="None", obs_mode=args.obs_mode,
        mode="ram", hud=False, render_oc_overlay=True,
        buffer_window_size=args.window, frameskip=args.frameskip,
        repeat_action_probability=0.25, full_action_space=False,
    )

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    wrapper_mapping = {
        "binary": ocatari_wrappers.BinaryMaskWrapper,
        "pixels": ocatari_wrappers.PixelMaskWrapper,
        "classes": ocatari_wrappers.ObjectTypeMaskWrapper,
        "planes": ocatari_wrappers.ObjectTypeMaskPlanesWrapper,
    }

    # If masked obs_mode are set, apply correct wrapper
    if args.masked_wrapper == "masked_dqn_bin":
        env = ocatari_wrappers.BinaryMaskWrapper(env, buffer_window_size=args.window,
                                                 )
    elif args.masked_wrapper == "masked_dqn_pixels":
        env = ocatari_wrappers.PixelMaskWrapper(env, buffer_window_size=args.window,
                                                )
    elif args.masked_wrapper == "masked_dqn_grayscale":
        env = ocatari_wrappers.ObjectTypeMaskWrapper(env, buffer_window_size=args.window,
                                                     )
    elif args.masked_wrapper == "masked_dqn_planes":
        env = ocatari_wrappers.ObjectTypeMaskPlanesWrapper(env, buffer_window_size=args.window,
                                                           )
    elif args.masked_wrapper == "masked_dqn_pixel_planes":
        env = ocatari_wrappers.PixelMaskPlanesWrapper(env, buffer_window_size=args.window,
                                                      )
    elif args.masked_wrapper == "masked_dl":
        env = ocatari_wrappers.DLWrapper(env, buffer_window_size=args.window,
                                         )
    elif args.masked_wrapper == "masked_dl_grouped":
        env = ocatari_wrappers.DLGroupedWrapper(env, buffer_window_size=args.window)
    elif args.masked_wrapper == "masked_dqn_planes_combined":
        env = ocatari_wrappers.ObjectTypeMaskPlanesWrapperCombined(env, buffer_window_size=args.window)
    elif args.masked_wrapper == "masked_dqn_planes_combined2":
        env = ocatari_wrappers.ObjectTypeMaskPlanesWrapperCombined2(env, buffer_window_size=args.window)
    elif args.masked_wrapper == "masked_dqn_bin_scaled":
        env = ocatari_wrappers.BinaryMaskWrapperScaled(env, buffer_window_size=args.window,
                                                       scale_w=args.scale_w,
                                                       scale_h=args.scale_h, keep_ratio=args.keep_ratio)
    elif args.masked_wrapper == "masked_dqn_pixels_scaled":
        env = ocatari_wrappers.PixelMaskWrapperScaled(env, buffer_window_size=args.window,
                                                      scale_w=args.scale_w,
                                                      scale_h=args.scale_h, keep_ratio=args.keep_ratio)
    elif args.masked_wrapper == "masked_dqn_grayscale_scaled":
        env = ocatari_wrappers.ObjectTypeMaskWrapperScaled(env, buffer_window_size=args.window,
                                                           scale_w=args.scale_w,
                                                           scale_h=args.scale_h, keep_ratio=args.keep_ratio)
    elif args.masked_wrapper == "masked_dqn_planes_scaled":
        env = ocatari_wrappers.ObjectTypeMaskPlanesWrapperScaled(env, buffer_window_size=args.window,

                                                                 scale_w=args.scale_w, scale_h=args.scale_h,
                                                                 keep_ratio=args.keep_ratio)
    elif args.masked_wrapper == "masked_dqn_pixel_planes_scaled":
        env = ocatari_wrappers.PixelMaskPlanesWrapperScaled(env, buffer_window_size=args.window,
                                                            scale_w=args.scale_w,
                                                            scale_h=args.scale_h, keep_ratio=args.keep_ratio)
    elif args.masked_wrapper == "masked_dqn_pixel_planes_scaled":
        env = ocatari_wrappers.PixelMaskPlanesWrapperScaled(env, buffer_window_size=args.window,
                                                            scale_w=args.scale_w,
                                                            scale_h=args.scale_h, keep_ratio=args.keep_ratio)
    elif args.masked_wrapper == "masked_dqn_parallelplanes":
        env = ocatari_wrappers.BigPlaneWrapper(
            env, buffer_window_size=args.window
        )
    elif args.masked_wrapper == "masked_dqn_bin_plus_og_obj":
        env = ocatari_wrappers.BinaryMaskWrapperPlusSimpleObj(env, buffer_window_size=args.window,)

    elif args.masked_wrapper == "masked_dqn_pixels_plus_og_obj":
        env = ocatari_wrappers.PixelMaskWrapperPlusSimpleObj(env, buffer_window_size=args.window,)
    elif args.masked_wrapper == "ext_obj":
        env = ocatari_wrappers.ObjExtended(env, use_distances=args.use_distances,
                                           use_direction=args.use_direction,
                                           multiply_player_info=args.multiply_player_info,
                                           use_angle=args.use_angle,
                                           use_overlap=args.use_overlap,

                                           use_object_xy=args.use_object_xy,
                                           double_input=args.double_input,
                                           apply_centerpoints=args.apply_centerpoints,
                                           use_origin_angle=args.use_origin_angle,
                                           use_origin_distances=args.use_origin_distances,
                                           use_vel=args.use_vel,
                                           use_time_distances=args.use_time_distances,
                                           use_time_angle=args.use_time_angle, overlap_offset=args.overlap_offset,
                                           )
        print("Observation space of extended Obj: " + str(env.observation_space))
    elif args.masked_wrapper != "":
        raise Exception(f"{args.masked_wrapper} not implemented.")
    if args.wrapper in wrapper_mapping:
        env = wrapper_mapping[args.wrapper](env)
    elif args.wrapper.endswith("+pixels"):
        base_wrapper = args.wrapper.split("+")[0]
        if base_wrapper in wrapper_mapping:
            env = wrapper_mapping[base_wrapper](env, include_pixels=True)

    is_combined = "plus" in args.masked_wrapper and "obj" in args.masked_wrapper
    if is_combined and args.obs_mode != "obj":
        raise Exception("obs_mode must be obj")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)
    results = {}
    for index, agent_path in enumerate(args.agents):

        print(agent_path)
        # if "rainbow" in agent_path.lower():
        #    agent, policy = load_agent_rainbow.load_agent(
        #        agent_path, env, "cpu")
        # else:
        if args.architecture == "PPO":
            if args.scale_w == 1.0 and args.scale_h == 1.0 and not args.keep_ratio:
                from cleanrl.architectures.ppo import PPODefault as Agent
                agent = Agent(env, device).to(device)
            else:
                print("PPOScaled is used!")
                from cleanrl.architectures.ppo import PPOScaled as Agent

                agent = Agent(env, device).to(device)
        elif args.architecture == "PPO_OBJ":
            from cleanrl.architectures.ppo import PPObj as Agent
            agent = Agent(env, device, args.encoder_dims, args.decoder_dims).to(device)
        elif args.architecture == "PPOCombi2Big":
            from cleanrl.architectures.ppo import PPOCombi2Big as Agent
            agent = Agent(env, device, args.encoder_dims, args.decoder_dims).to(device)
        else:
            raise NotImplementedError(f"Architecture {args.architecture} does not exist!")

        model_path = agent_path
        print(f"Path: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_weights'])
        agent.eval()
        policy = lambda x: agent.get_action_and_value(x)[0]
        print(f"Loaded agent from {agent_path}")

        rewards = []
        for episode in range(args.episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if is_combined:
                    obs_m = obs["masks"]
                    obs_o = obs["obj"]
                    obs_m_tensor = torch.Tensor(obs_m).to(device)
                    obs_o_tensor = torch.Tensor(obs_o).to(device)
                    obs_tensor = {"masks": obs_m_tensor.unsqueeze(0), "obj": obs_o_tensor.unsqueeze(0)}
                    action = policy(obs_tensor)[0]
                else:
                    obs = torch.Tensor(obs).to(device)
                    action = policy(obs.unsqueeze(0))[0]
                obs, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        median_reward = np.median(rewards)
        hns_reward = calculate_hns(avg_reward, game_name)
        iqm_reward = rlm.aggregate_iqm(np.array(rewards))
        hns_iqm = calculate_hns(iqm_reward, game_name)
        import time
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        run_name = f"{args.game}_{current_time}_{agent_path}_{index}"

        results[run_name] = {
            "model": agent_path,
            "episode_rewards": rewards,
            "mean_reward": avg_reward,
            "std_reward": std_reward,
            "median_reward": median_reward,
            "iqm_reward": iqm_reward,
            "hns_mean": hns_reward,
            "hns_iqm": hns_iqm,
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "human": atari_scores[game_name.lower()][1],
            "random": atari_scores[game_name.lower()][0],
            "Args": vars(args),
        }
        print(f"\nSummary for {agent_path}:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Standard Deviation: {std_reward:.2f}")
        print(f"Median Reward: {median_reward:.2f}")
        print(f"HNS: {hns_reward:.2f}")
        print(f"IQM Reward: {iqm_reward:.2f}")
        print(f"HNS (IQM): {hns_iqm:.2f}")
        print("--------------------------------------")

    try:
        with open(args.output, "r") as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []

    if existing_results is None:
        existing_results = []

    existing_results.append(results)

    with open(args.output, "w") as f:
        json.dump(existing_results, f, indent=4)

    env.close()


if __name__ == "__main__":
    main()
