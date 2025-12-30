# script for comparing the object focus of agents using different input representations

# Methods to aggregate the channels
aggregation_methods = ["mean", "max", ]
# When creating a video, this method is used for aggregating the buffer window
aggregation_method_window_corr = aggregation_methods[0]
# When computing the object focus, this method is used for aggregating the planes of the different objects
aggregation_method_planes_corr = aggregation_methods[0]

# If toggled, the extended comparison masks are shown in the videos. Otherwise, the non-extended masks are shown.
show_extendend_in_video = False
# If toggled, computes the correlations between the saliency maps and the comparison masks
compute_corr = True
# If toggled, the results of the object focus comparisons are stored in json files
write_json = True

compute_correlation_between_action_and_value = False

from scipy import stats
from hackatari import HackAtari, HumanPlayable
import numpy as np
import cv2
import torch
import time
import argparse
import json

from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
import rliable.metrics as rlm
from captum.attr import Saliency, GuidedBackprop, GuidedGradCam, IntegratedGradients
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ocatari_wrappers
from reference_scores.scores_for_hns import atari_scores

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


def compute_corr_coefficients(imgs_value_list, imgs_policy_list, imgs_input_list, tmp_results, ):
    """
    Compute correlations between the saliency maps and the comparison masks
    """
    imgs_value_array = np.stack(imgs_value_list, axis=0)
    imgs_policy_array = np.stack(imgs_policy_list, axis=0)
    imgs_mask_array = np.stack(imgs_input_list, axis=0)
    n_timesteps = imgs_value_array.shape[0]
    imgs_values_flat = imgs_value_array.reshape(n_timesteps, -1)
    imgs_policy_flat = imgs_policy_array.reshape(n_timesteps, -1)
    imgs_mask_flat = imgs_mask_array.reshape(n_timesteps, -1)

    corr_mask_value = np.corrcoef(imgs_values_flat.ravel(), imgs_mask_flat.ravel())[0, 1]
    corr_mask_policy = np.corrcoef(imgs_policy_flat.ravel(), imgs_mask_flat.ravel())[0, 1]
    print(f"Correlation between mask and value: {corr_mask_value}")
    print(f"Correlation between mask and policy: {corr_mask_policy}")
    tmp_results["corr_mask_value"].append(float(corr_mask_value))
    tmp_results["corr_mask_policy"].append(float(corr_mask_policy))

    res = stats.spearmanr(imgs_values_flat.ravel(), imgs_mask_flat.ravel())
    corr_mask_value2 = res.statistic
    res = stats.spearmanr(imgs_policy_flat.ravel(), imgs_mask_flat.ravel())
    corr_mask_policy2 = res.statistic
    tmp_results["corr_mask_value2"].append(float(corr_mask_value2))
    tmp_results["corr_mask_policy2"].append(float(corr_mask_policy2))
    print(f"2. Correlation between mask and value: {corr_mask_value2}")
    print(f"2. Correlation between mask and policy: {corr_mask_policy2}")


def aggregate_channels(obs, imgs_value, imgs_policy, aggregation_method):
    if aggregation_method == "max":
        image = np.max(obs, axis=0)
        v = np.max(imgs_value, axis=0)
        p = np.max(imgs_policy, axis=0)
    elif aggregation_method == "mean":
        image = np.mean(obs, axis=0)
        v = np.mean(imgs_value, axis=0)
        p = np.mean(imgs_policy, axis=0)
    else:
        raise ValueError(aggregation_method)
    return image, v, p


def aggregate_buffer_window_planes(obs, imgs_value, imgs_policy, aggregation_method, window, num_obj_total):
    reshaped_imgs_value = imgs_value.reshape(window, num_obj_total, *obs.shape[1:])
    reshaped_imgs_policy = imgs_policy.reshape(window, num_obj_total, *obs.shape[1:])
    reshaped_obs = obs.reshape(window, num_obj_total, *obs.shape[1:])
    obs2, imgs_value2, imgs_policy2 = aggregate_channels(reshaped_obs, reshaped_imgs_value, reshaped_imgs_policy,
                                                         aggregation_method)
    return obs2, imgs_value2, imgs_policy2


def fill_lists_planes(window, num_obj_to_use, num_obj_total, obs, imgs_value, imgs_policy, imgs_input_list,
                      imgs_value_list, imgs_policy_list, ):
    obs2, imgs_value2, imgs_policy2 = aggregate_buffer_window_planes(obs, imgs_value, imgs_policy,
                                                                     aggregation_method_window_corr, window,
                                                                     num_obj_total)
    imgs_policy3 = imgs_policy2[0: num_obj_to_use]
    imgs_value3 = imgs_value2[0: num_obj_to_use]
    obs3 = obs2[0: num_obj_to_use]
    image, v, p = aggregate_channels(obs3, imgs_value3, imgs_policy3, aggregation_method_planes_corr)

    if image.sum() != 0:
        imgs_input_list.append(image / image.sum())
    else:
        imgs_input_list.append(np.zeros_like(image))

    if v.sum() != 0:
        imgs_value_list.append(v / v.sum())
    else:
        imgs_value_list.append(np.zeros_like(v))

    if p.sum() != 0:
        imgs_policy_list.append(p / p.sum())
    else:
        imgs_policy_list.append(np.zeros_like(p))


# For each input representation, this dictionary contains the version with the enlarged bounding boxes
extensions_dict = {
    "planes": "planes_extended",
    "binary_mask": "binary_mask_extended",
    "parallel_planes": "parallel_planes_extended",
    "pixel_screen": "binary_mask_extended",
}


def fill_lists(obs, imgs_value, imgs_policy, imgs_input_list, imgs_value_list, imgs_policy_list):
    mask_img, img_value, img_policy = aggregate_channels(obs, imgs_value, imgs_policy,
                                                         aggregation_method_window_corr)

    if img_value.sum() != 0:
        imgs_value_list.append(img_value / img_value.sum())
    else:
        imgs_value_list.append(np.zeros_like(img_value))

    if img_policy.sum() != 0:
        imgs_policy_list.append(img_policy / img_policy.sum())
    else:
        imgs_policy_list.append(np.zeros_like(img_policy))

    if mask_img.sum() != 0:
        imgs_input_list.append(mask_img / mask_img.sum())
    else:
        imgs_input_list.append(np.zeros_like(mask_img))


def compute_saliency(agent, captum_algo, obs_t):
    def critic_output(x):
        return agent.get_value(x)

    def actor_output(x):
        _, l, _, _ = agent.get_action_and_value(x)
        return l

    agent.zero_grad()
    if captum_algo == "GuidedBackprop":
        value_saliency = GuidedBackprop(agent)
        value_grad = value_saliency.attribute(obs_t, additional_forward_args="critic").clone()
        agent.zero_grad()

        policy_saliency = GuidedBackprop(agent)
        policy_grad = policy_saliency.attribute(obs_t, additional_forward_args="actor").clone()
    elif captum_algo == "Saliency":
        value_saliency = Saliency(critic_output)
        value_grad = value_saliency.attribute(obs_t).clone()

        agent.zero_grad()
        policy_saliency = Saliency(actor_output)
        policy_grad = policy_saliency.attribute(obs_t).clone()
    else:
        raise NotImplementedError
    return value_grad, policy_grad


def calculate_hns(score, game):
    human_score = atari_scores[game.lower()][1]
    random_score = atari_scores[game.lower()][0]
    return (score - random_score) / (human_score - random_score)


def main():
    parser = HackAtariArgumentParser(description="HackAtari Experiment Runner")
    parser.add_argument("-g", "--game", type=str,
                        default="Seaquest", help="Game to be run")
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
                        required=False, help="List of trained agent model paths")
    parser.add_argument("-mo", "--game_mode", type=int,
                        default=0, help="Alternative ALE game mode")
    parser.add_argument("-d", "--difficulty", type=int,
                        default=0, help="Alternative ALE difficulty")
    parser.add_argument("-e", "--episodes", type=int,
                        default=1, help="Number of episodes per agent")
    parser.add_argument("-out", "--output", type=str,
                        default="results.json", help="Output file for results")
    parser.add_argument("-ar", "--architecture", type=str,
                        default="PPO", help="")
    parser.add_argument("-cu", "--cuda", type=bool,
                        default=True, help="")

    parser.add_argument("-n", "--name", type=str,
                        default="", help="Name of agent")

    parser.add_argument("-mw", "--masked_wrapper", type=str,
                        default="",
                        help="Input representation that the agent receives. Does not have to be one of OCCAM's wrappers, can also be obs_mode_dqn")

    parser.add_argument("-ext", "--extension", type=int, default=0,
                        help="By many pixels should each object's bounding box be extended in each direction")
    parser.add_argument("-uh", "--use_hud", action='store_true')

    parser.add_argument("-ca", "--captum_algo", type=str,
                        default="Saliency", help="Which model interpretability method from Captum should be used?")

    parser.add_argument("-am", "--aggregation_method", type=str, default="default",
                        help="How should the different channels be aggregated?")

    args = parser.parse_args()

    if args.masked_wrapper == "masked_dqn_planes":
        mw = "planes"
    elif args.masked_wrapper == "masked_dqn_bin":
        mw = "binary_mask"
    elif args.masked_wrapper == "masked_dqn_parallelplanes" or args.masked_wrapper == "parallelplanes":
        mw = "parallel_planes"
    elif args.masked_wrapper == "obs_mode_dqn":
        mw = "pixel_screen"
    else:
        print(args.masked_wrapper)
        raise NotImplementedError

    if args.aggregation_method != "default":
        print(args.aggregation_method)
        global aggregation_method_window_video, aggregation_method_window_corr, aggregation_method_planes_video, aggregation_method_planes_corr
        if args.aggregation_method in aggregation_methods:
            aggregation_method_window_video = args.aggregation_method
            aggregation_method_window_corr = args.aggregation_method
            aggregation_method_planes_video = args.aggregation_method
            aggregation_method_planes_corr = args.aggregation_method
        else:
            raise NotImplementedError

    s = args.game
    if s.endswith("-v5"):
        s = s[:-3]
    if s.startswith("ALE/"):
        s = s[len("ALE/"):]
    game_name = s
    if not args.game.startswith("ALE/") and not args.game.endswith("-v5"):
        args.game = args.game[0].upper() + s[1:]
    if args.game == "spaceinvaders" or args.game == "Spaceinvaders":
        args.game = "SpaceInvaders"
    env = HackAtari(
        args.game, args.modifs, args.reward_function,
        dopamine_pooling=args.dopamine_pooling, game_mode=args.game_mode,
        difficulty=args.difficulty, render_mode="None", obs_mode="dqn",
        mode="ram", hud=False, render_oc_overlay=True,
        buffer_window_size=args.window, frameskip=args.frameskip,
        repeat_action_probability=0.25, full_action_space=False,
    )

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = ocatari_wrappers.MaskedBaseWrapperDict(env, buffer_window_size=args.window, extension=args.extension,
                                                        work_in_output_shape_planes=False,
                                                        work_in_output_shape_binary_mask=False)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)
    results = {}
    from collections import defaultdict
    tmp_results = defaultdict(list)

    for idx, agent_path in enumerate(args.agents):
        print(agent_path)
        from cleanrl.architectures.ppo import PPODefault as Agent
        dims = env.observation_space[mw].shape
        agent = Agent(env, device, dims).to(device)

        model_path = agent_path
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_weights'])
        # agent.eval()

        policy = lambda x: agent.get_action_and_value(x)
        print(f"Loaded agent from {agent_path}")

        if args.masked_wrapper == "masked_dqn_planes":
            num_obj_without_hud = ocatari_wrappers.ObjectTypeMaskPlanesWrapper(env, buffer_window_size=1,
                                                                               v2=True).observation_space.shape[
                0]

            num_obj_total = int(env.observation_space["planes"].shape[0] / args.window)
            num_obj_to_use = num_obj_total if args.use_hud else num_obj_without_hud

        rewards = []

        imgs_value_list = []
        imgs_policy_list = []
        imgs_input_list = []

        for episode in range(args.episodes):
            count = 0
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            obs_wrapper = obs[mw]
            obs_t = torch.Tensor(obs_wrapper).unsqueeze(0).to(device)
            while not done:
                count += 1
                action = policy(obs_t)[0]

                obs, reward, terminated, truncated, _ = env.step(action)
                obs_wrapper = obs[mw]
                obs_extended = obs[extensions_dict[mw]]

                obs_t = torch.Tensor(obs_wrapper).unsqueeze(0).to(device)

                obs_t.requires_grad_(True)

                value_grad, policy_grad = compute_saliency(agent, args.captum_algo, obs_t)

                imgs_policy = np.abs(policy_grad.squeeze().cpu().data.numpy())
                imgs_value = np.abs(value_grad.squeeze().cpu().data.numpy())
                if args.masked_wrapper == "masked_dqn_planes":
                    if compute_corr:
                        fill_lists_planes(args.window, num_obj_to_use, num_obj_total, obs_extended, imgs_value,
                                          imgs_policy,
                                          imgs_input_list,
                                          imgs_value_list,
                                          imgs_policy_list, )
                else:
                    if compute_corr:
                        fill_lists(obs_extended, imgs_value, imgs_policy, imgs_input_list, imgs_value_list,
                                   imgs_policy_list)

                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            print(f"Episode {episode + 1}: Length = {count}")

        if compute_corr:
            print("=================================")
            print("For the mask input:")
            compute_corr_coefficients(imgs_value_list, imgs_policy_list, imgs_input_list, tmp_results)

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        median_reward = np.median(rewards)
        hns_reward = calculate_hns(avg_reward, game_name)
        iqm_reward = rlm.aggregate_iqm(np.array(rewards))
        hns_iqm = calculate_hns(iqm_reward, game_name)

        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        run_name = f"{args.game}_{current_time}_{model_path}"

        results[run_name] = {
            "model": model_path,
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
        results[run_name]["metrics"] = tmp_results

    results["metrics"] = tmp_results

    if write_json:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(args.output, "r") as f:
                existing_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = {}

        if args.name not in existing_results:
            existing_results[args.name] = {}
        existing_results[args.name][args.game.lower()] = results

        with open(args.output, "w") as f:
            json.dump(existing_results, f, indent=4)

    env.close()


if __name__ == "__main__":
    main()
