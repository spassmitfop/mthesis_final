#script for training agents
import os
import sys
import tyro
import time
import random
import warnings

import numpy as np

from tqdm import tqdm
from rtpt import RTPT
from pathlib import Path
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from typing import Literal

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ocatari_wrappers

# Suppress warnings to avoid cluttering output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set CUDA environment variable for determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add custom paths if OC_ATARI_DIR is set (optional integration for extended functionality)
oc_atari_dir = os.getenv("OC_ATARI_DIR")
if oc_atari_dir is not None:
    oc_atari_path = os.path.join(Path(__file__), oc_atari_dir)
    sys.path.insert(1, oc_atari_path)

# Command line argument configuration using dataclass
@dataclass
class Args:
    # General
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Environment parameters
    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    obs_mode: str = "dqn"
    """observation mode for OCAtari"""
    feature_func: str = ""
    """the object features to use as observations"""
    buffer_window_size: int = 4
    """length of history in the observations"""
    backend: str = "OCAtari"
    """Which Backend should we use"""
    modifs: str = ""
    """Modifications for Hackatari"""
    new_rf: str = ""
    """Path to a new reward functions for OCALM and HACKATARI"""
    frameskip: int = 4
    """the frame skipping option of the environment"""

    # Tracking (Logging and monitoring configurations)
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "OC-Transformer"
    """the wandb's project name"""
    wandb_entity: str = "AIML_OC"
    """the entity (team) of wandb's project"""
    wandb_dir: str = None
    """the wandb directory"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    ckpt: str = ""
    """Path to a checkpoint to a model to start training from"""
    logging_level: int = 40
    """Logging level for the Gymnasium logger"""
    author: str = "JB"
    """Initials of the author"""
    checkpoint_interval: int = 40
    """Number of iterations before a model checkpoint is saved and uploaded to wandb"""

    # Algorithm-specific arguments
    architecture: str = "PPO"
    """ Specifies the used architecture"""

    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer parameters
    emb_dim: int = 128
    """input embedding size of the transformer"""
    num_heads: int = 64
    """number of multi-attention heads"""
    num_blocks: int = 1
    """number of transformer blocks"""
    patch_size: int = 12
    """ViT patch size"""

    # PPObj network parameters
    encoder_dims: list[int] = (256, 512, 1024, 512)
    """layer dimensions before nn.Flatten()"""
    decoder_dims: list[int] = (512,)
    """layer dimensions after nn.Flatten()"""

    # HackAtari testing
    test_modifs: str = ""
    """Modifications for Hackatari"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    masked_wrapper: str = ""
    """the obs_mode if a masking wrapper is needed (set in runtime)"""
    add_pixels: bool = False
    """should the grayscale game screen be added to the observations (set in runtime)"""
    use_distances: bool = False
    """adds the distance feature to ext_obj"""
    use_direction: bool = False
    """adds the direction feature to ext_obj """
    multiply_player_info: bool = False
    """adds the multiply player info feature to ext_obj """
    use_angle: bool = False
    """adds the angle feature to ext_obj """
    use_overlap: bool = False
    """adds the overlap feature to ext_obj """
    use_object_xy: bool = True
    """should ext_obj include each object's xy-coordinates?"""
    scale_w: float = 1.0
    """scale factor for the width"""
    scale_h: float = 1.0
    """scale factor for the height"""
    keep_ratio: bool = False
    """should the original aspect ratio of the game screen (210:160) be kept?"""
    double_input: bool = False
    """doubles the ext_obj x by concatenating it with itself: [x,x]"""
    apply_centerpoints: bool = False
    """should the center point of each object be used to compute the feautures?"""
    use_origin_angle: bool = False
    """adds the angle feature to ext_obj, with the player's position replaced by the origin"""
    use_origin_distances: bool = False
    """adds the distance feature to ext_obj, with the player's position replaced by the origin"""
    use_vel: bool = False
    """adds the velocity feature to ext_obj"""
    use_time_angle: bool = False
    """adds the time angle feature to ext_obj, the angle between each object's current and last position"""
    use_time_distances: bool = False
    """adds the time distance feature to ext_obj, the distance between each object's current and last position"""
    overlap_offset: int = 2
    """When detecting overlaps, the bounding boxes are increased by this number"""
    base_dir: str = "shared"
    """directory in which the agent is saved"""

# Global variable to hold parsed arguments
global args


# -----------------------
# Helpers
# -----------------------
def seed_everything(seed: int, cuda: bool = True, torch_deterministic: bool = True):
    """Seed python, numpy, torch (+cuda) and set deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(torch_deterministic)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = False


# Function to create a gym environment with the specified settings
def make_env(env_id, idx, seed, capture_video, run_dir):
    """
    Creates a gym environment with the specified settings.
    """

    def thunk():
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        logger.set_level(args.logging_level)
        # Setup environment based on backend type (HackAtari, OCAtari, Gym)
        if args.backend == "HackAtari":
            print("Hacker man")
            from hackatari.core import HackAtari
            modifs = [i for i in args.modifs.split(" ") if i]
            env = HackAtari(
                env_id,
                modifs=modifs,
                rewardfunc_path=args.new_rf,
                obs_mode=args.obs_mode,
                hud=False,
                render_mode="rgb_array",
                frameskip=args.frameskip,
                create_buffer_stacks=[]
            )
        elif args.backend == "OCAtari":
            from ocatari.core import OCAtari
            env = OCAtari(
                env_id,
                hud=False,
                render_mode="rgb_array",
                obs_mode=args.obs_mode,
                frameskip=args.frameskip,
                create_buffer_stacks=[]
            )
        elif args.backend == "Gym":
            # Use Gym backend with image preprocessing wrappers
            env = gym.make(env_id, render_mode="rgb_array", frameskip=args.frameskip)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, args.buffer_window_size)
        else:
            raise ValueError("Unknown Backend")

        # Capture video if required
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env,
                                           f"{run_dir}/media/videos",
                                           disable_logger=True)

        # Apply standard Atari environment wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        print("Masked wrapper: " + str(args.masked_wrapper))
        # If architecture is OCT, apply OCWrapper to environment
        if args.architecture == "OCT":
            from ocrltransformer.wrappers import OCWrapper
            env = OCWrapper(env)

        # If masked obs_mode are set, apply correct wrapper
        elif args.masked_wrapper == "masked_dqn_bin":
            env = ocatari_wrappers.BinaryMaskWrapper(env, buffer_window_size=args.buffer_window_size,
                                                     include_pixels=args.add_pixels)
        elif args.masked_wrapper == "masked_dqn_pixels":
            env = ocatari_wrappers.PixelMaskWrapper(env, buffer_window_size=args.buffer_window_size,
                                                    include_pixels=args.add_pixels)
        elif args.masked_wrapper == "masked_dqn_grayscale":
            env = ocatari_wrappers.ObjectTypeMaskWrapper(env, buffer_window_size=args.buffer_window_size,
                                                         include_pixels=args.add_pixels)
        elif args.masked_wrapper == "masked_dqn_planes":
            env = ocatari_wrappers.ObjectTypeMaskPlanesWrapper(env, buffer_window_size=args.buffer_window_size,
                                                               include_pixels=args.add_pixels)
        elif args.masked_wrapper == "masked_dqn_pixel_planes":
            env = ocatari_wrappers.PixelMaskPlanesWrapper(env, buffer_window_size=args.buffer_window_size,
                                                          include_pixels=args.add_pixels)
        elif args.masked_wrapper == "masked_dl":
            env = ocatari_wrappers.DLWrapper(env, buffer_window_size=args.buffer_window_size,
                                             include_pixels=args.add_pixels)
        elif args.masked_wrapper == "masked_dl_grouped":
            env = ocatari_wrappers.DLGroupedWrapper(env, buffer_window_size=args.buffer_window_size)
        elif args.masked_wrapper == "masked_dqn_planes_combined":
            env = ocatari_wrappers.ObjectTypeMaskPlanesWrapperCombined(env, buffer_window_size=args.buffer_window_size)
        elif args.masked_wrapper == "masked_dqn_planes_combined2":
            env = ocatari_wrappers.ObjectTypeMaskPlanesWrapperCombined2(env, buffer_window_size=args.buffer_window_size)

        elif args.masked_wrapper == "masked_dqn_bin_scaled":
            env = ocatari_wrappers.BinaryMaskWrapperScaled(env, buffer_window_size=args.buffer_window_size,
                                                           include_pixels=args.add_pixels, scale_w=args.scale_w,
                                                           scale_h=args.scale_h, keep_ratio=args.keep_ratio)
        elif args.masked_wrapper == "masked_dqn_pixels_scaled":
            env = ocatari_wrappers.PixelMaskWrapperScaled(env, buffer_window_size=args.buffer_window_size,
                                                          include_pixels=args.add_pixels, scale_w=args.scale_w,
                                                          scale_h=args.scale_h, keep_ratio=args.keep_ratio)
        elif args.masked_wrapper == "masked_dqn_grayscale_scaled":
            env = ocatari_wrappers.ObjectTypeMaskWrapperScaled(env, buffer_window_size=args.buffer_window_size,
                                                               include_pixels=args.add_pixels, scale_w=args.scale_w,
                                                               scale_h=args.scale_h, keep_ratio=args.keep_ratio)
        elif args.masked_wrapper == "masked_dqn_planes_scaled":
            env = ocatari_wrappers.ObjectTypeMaskPlanesWrapperScaled(env, buffer_window_size=args.buffer_window_size,
                                                                     include_pixels=args.add_pixels,
                                                                     scale_w=args.scale_w, scale_h=args.scale_h,
                                                                     keep_ratio=args.keep_ratio)
        elif args.masked_wrapper == "masked_dqn_pixel_planes_scaled":
            env = ocatari_wrappers.PixelMaskPlanesWrapperScaled(env, buffer_window_size=args.buffer_window_size,
                                                                include_pixels=args.add_pixels, scale_w=args.scale_w,
                                                                scale_h=args.scale_h, keep_ratio=args.keep_ratio)
        elif args.masked_wrapper == "masked_dqn_parallelplanes":
            env = ocatari_wrappers.BigPlaneWrapper(
                env, buffer_window_size=args.buffer_window_size, include_pixels=args.add_pixels
            )

        elif args.masked_wrapper == "masked_dqn_bin_plus_og_obj":
            env = ocatari_wrappers.BinaryMaskWrapperPlusSimpleObj(env, buffer_window_size=args.buffer_window_size,
                                                                  include_pixels=args.add_pixels)

        elif args.masked_wrapper == "masked_dqn_pixels_plus_og_obj":
            env = ocatari_wrappers.PixelMaskWrapperPlusSimpleObj(env, buffer_window_size=args.buffer_window_size,
                                                                 include_pixels=args.add_pixels)
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
                                               use_time_angle=args.use_time_angle, use_time_distances=args.use_time_distances,
                                               overlap_offset=args.overlap_offset,
                                               )
            print("Observation space of extended Obj: " + str(env.observation_space))
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    # Parse command-line arguments using Tyro
    args = tyro.cli(Args)
    # Compute runtime-dependent arguments
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    is_combined = "plus" in args.masked_wrapper and "obj" in args.masked_wrapper

    # prepare for masking wrappers
    if "masked" in args.obs_mode:
        # import ocatari_wrappers

        if args.obs_mode.endswith("+pixels"):
            args.masked_wrapper = args.obs_mode[:-7]
            args.add_pixels = True
        else:
            args.masked_wrapper = args.obs_mode
            args.add_pixels = False
        args.obs_mode = "ori"
    seed_everything(args.seed, cuda=args.cuda, torch_deterministic=args.torch_deterministic)

    # Generate run name based on environment, experiment, seed, and timestamp
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Initialize tracking with Weights and Biases if enabled
    if args.track:
        import dataclasses, wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=dataclasses.asdict(args),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=args.wandb_dir,
            job_type="train",
            group=f"{args.env_id}_{args.architecture}",
            tags=[args.env_id, args.architecture, args.backend, args.obs_mode],
            resume="allow",
        )
        wandb.define_metric("global_step")
        wandb.define_metric("charts/*", step_metric="global_step")
        wandb.define_metric("losses/*", step_metric="global_step")
        wandb.define_metric("time/*", step_metric="global_step")
        writer_dir = run.dir
        postfix = dict(url=run.url)
    else:
        writer_dir = f"{args.wandb_dir}/runs/{run_name}"
        postfix = None

    # Initialize Tensorboard SummaryWriter to log metrics and hyperparameters
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create RTPT object to monitor progress with estimated time remaining
    rtpt = RTPT(name_initials=args.author, experiment_name=args.exp_name,
                max_iterations=args.num_iterations)
    rtpt.start()  # Start RTPT tracking

    # Set logger level and determine whether to use GPU or CPU for computation
    logger.set_level(args.logging_level)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.debug(f"Using device {device}.")

    # Environment setup
    envs = SubprocVecEnv(
        [make_env(args.env_id, i, args.seed + i, args.capture_video, writer_dir) for i in range(args.num_envs)]
    )
    envs = VecNormalize(envs, norm_obs=False, norm_reward=True)

    # Define the agent's architecture based on command line arguments
    if args.architecture == "OCT":
        from architectures.transformer import OCTransformer as Agent

        agent = Agent(envs, args.emb_dim, args.num_heads, args.num_blocks, device).to(device)
    elif args.architecture == "VIT":
        from architectures.transformer import VIT as Agent

        agent = Agent(envs, args.emb_dim, args.num_heads, args.num_blocks,
                      args.patch_size, args.buffer_window_size, device).to(device)
    elif args.architecture == "VIT2":
        from architectures.transformer import SimpleViT2 as Agent

        agent = Agent(envs, args.emb_dim, args.num_heads, args.num_blocks,
                      args.patch_size, args.buffer_window_size, device).to(device)
    elif args.architecture == "MobileVit":
        from architectures.transformer import MobileVIT as Agent

        agent = Agent(envs, args.emb_dim, device).to(device)
    elif args.architecture == "MobileVit2":
        from architectures.transformer import MobileViT2 as Agent

        agent = Agent(envs, args.emb_dim, args.num_heads, args.num_blocks,
                      args.patch_size, args.buffer_window_size, device).to(device)
    elif args.architecture == "PPO":
        if args.scale_w == 1.0 and args.scale_h == 1.0 and not args.keep_ratio:
            from architectures.ppo import PPODefault as Agent

            agent = Agent(envs, device).to(device)
        else:
            print("PPOScaled is used!")
            from architectures.ppo import PPOScaled as Agent

            agent = Agent(envs, device).to(device)
    elif args.architecture == "PPO_OBJ":
        from architectures.ppo import PPObj as Agent

        agent = Agent(envs, device, args.encoder_dims, args.decoder_dims).to(device)
        # agent = Agent(envs=envs, device=device, encoder_dims=args.encoder_dims, decoder_dims=args.decoder_dims, double_input=args.double_input).to(device)
    elif args.architecture == "PPOCombi2Big":
        from architectures.ppo import PPOCombi2Big as Agent

        agent = Agent(envs, device, args.encoder_dims, args.decoder_dims).to(device)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} does not exist!")

    # Initialize optimizer for training
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.track:
        num_params = sum(p.numel() for p in agent.parameters())
        wandb.summary["params_total"] = num_params
        wandb.watch(agent, log="gradients", log_freq=1000, log_graph=False)

    global_step = 0
    start_time = time.time()
    act_space_shape = envs.action_space.shape
    if is_combined:
        obs_m = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space["masks"].shape,
                            dtype=torch.float32, device=device)
        obs_o = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space["obj"].shape, dtype=torch.float32,
                            device=device)

        next_obs_dict = envs.reset()
        next_obs_m = next_obs_dict["masks"]
        next_obs_o = next_obs_dict["obj"]
        next_obs_m = torch.tensor(next_obs_m, dtype=torch.float32, device=device)
        next_obs_o = torch.tensor(next_obs_o, dtype=torch.float32, device=device)
        next_obs = {"masks": next_obs_m, "obj": next_obs_o}
    else:
        # Allocate rollout storage (clear dtypes, on device)
        obs_space_shape = envs.observation_space.shape
        obs = torch.zeros((args.num_steps, args.num_envs) + obs_space_shape, dtype=torch.float32, device=device)

        next_obs = envs.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space_shape, dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    pbar = tqdm(range(1, args.num_iterations + 1), postfix=postfix)
    for iteration in pbar:
        # LR anneal
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Episode collectors / aggregates
        episode_returns, episode_lengths = [], []
        elength = 0
        eorgr = 0.0
        enewr = 0.0
        count = 0
        # Checkpoint
        ''' 
        if iteration % args.checkpoint_interval == 0:
            model_path = f"{writer_dir}/{args.exp_name}_{iteration}.cleanrl_model"
            model_data = {
                "model_weights": agent.state_dict(),
                "args": vars(args),
                "Timesteps": iteration * args.batch_size
            }
            torch.save(model_data, model_path)
            logger.info(f"model saved to {model_path} at iteration {iteration}")
            if args.track:
                _log_model_artifact(
                    run, model_path, name=f"{args.exp_name}",
                    iteration=iteration, metadata={"env": args.env_id, "seed": args.seed}
                )
        '''

        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            if is_combined:
                obs_o[step] = next_obs_o
                obs_m[step] = next_obs_m
            else:
                obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, next_done_np, infos = envs.step(action.detach().cpu().numpy())
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device).view(-1)
            if is_combined:
                next_obs_m = torch.tensor(next_obs_np["masks"], dtype=torch.float32, device=device)
                next_obs_o = torch.tensor(next_obs_np["obj"], dtype=torch.float32, device=device)
                next_obs = {"masks": next_obs_m, "obj": next_obs_o}
            else:
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)
            # Per-episode stats
            if bool(next_done.bool().any()):
                for info in infos:
                    if "episode" in info:
                        r = float(info["episode"]["r"])
                        l = int(info["episode"]["l"])
                        episode_returns.append(r)
                        episode_lengths.append(l)
                        count += 1
                        elength += l
                        if args.new_rf:
                            enewr += r
                            eorgr += float(info.get("org_return", 0.0))
                        else:
                            eorgr += r

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # Flatten
        if is_combined:
            b_obs_m = obs_m.reshape((-1,) + envs.observation_space["masks"].shape)
            b_obs_o = obs_o.reshape((-1,) + envs.observation_space["obj"].shape)
        else:
            b_obs = obs.reshape((-1,) + obs_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy & value
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                if is_combined:
                    b_obs_here = b_obs_dict = {"masks": b_obs_m[mb_inds], "obj": b_obs_o[mb_inds]}
                else:
                    b_obs_here = b_obs[mb_inds]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs_here, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.track and (start % (args.minibatch_size * 4) == 0):
                    import wandb

                    wandb.log({"losses/grad_total_norm": float(gn)}, step=global_step)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Explained variance
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TensorBoard scalars (remain)
        if count > 0:
            if args.new_rf:
                writer.add_scalar("charts/Episodic_New_Reward", enewr / count, global_step)
            writer.add_scalar("charts/Episodic_Original_Reward", eorgr / count, global_step)
            writer.add_scalar("charts/Episodic_Length", elength / count, global_step)
            pbar.set_description(f"Reward: {eorgr / count:.1f}")

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # W&B enrichments
        if args.track:
            import wandb

            log_payload = {
                "global_step": global_step,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": float(np.mean(clipfracs)),
                "losses/explained_variance": float(explained_var),
                "time/SPS": int(global_step / (time.time() - start_time)),
            }
            if count > 0:
                if args.new_rf:
                    log_payload["charts/Episodic_New_Reward"] = enewr / count
                log_payload["charts/Episodic_Original_Reward"] = eorgr / count
                log_payload["charts/Episodic_Length"] = elength / count
            if episode_returns:
                log_payload["charts/ReturnHist"] = wandb.Histogram(episode_returns)
                log_payload["charts/LengthHist"] = wandb.Histogram(episode_lengths)
            if device.type == "cuda":
                log_payload["sys/gpu_mem_alloc_GB"] = torch.cuda.memory_allocated() / 1e9
                log_payload["sys/gpu_mem_reserved_GB"] = torch.cuda.memory_reserved() / 1e9
            wandb.log(log_payload, step=global_step)

        rtpt.step()

    last_part = args.env_id.split('/')[-1]
    game_name = last_part.split('-')[0]
    model_path = os.path.join(
        *[args.base_dir, game_name.lower(), str(args.seed), f"{args.exp_name}.cleanrl_model"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_data = {
        "model_weights": agent.state_dict(),
        "args": vars(args),
    }
    torch.save(model_data, model_path)
    logger.info(f"model saved to {model_path} in epoch {epoch}")

    # Log final model and performance with Weights and Biases if enabled
    if args.track:
        wandb.log({"model_path": model_path})
        # Log model to Weights and Biases
        name = f"{args.exp_name}_s{args.seed}"
        # run.log_model(model_path, name)  # noqa: cannot be undefined

        # Evaluate agent's performance
        args.new_rf = ""
        from typing import Callable


        def evaluate_local(
                agent,
                make_env: Callable,
                eval_episodes: int,
                env_id,
                capture_video,
                run_dir,
                device
        ):
            envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, args.seed, capture_video, run_dir)])
            agent.eval()

            obs, _ = envs.reset()
            if is_combined:
                obs_m = torch.Tensor(obs["masks"]).to(device)
                obs_o = torch.Tensor(obs["obj"]).to(device)
                obs_t = {"masks": obs_m, "obj": obs_o}
            else:
                obs_t = torch.Tensor(obs).to(device)
            episodic_returns = []
            while len(episodic_returns) < eval_episodes:
                actions, _, _, _ = agent.get_action_and_value(obs_t)
                next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if "episode" not in info:
                            continue
                        episodic_returns += [info["episode"]["r"]]
                obs = next_obs
                if is_combined:
                    obs_m = torch.Tensor(obs["masks"]).to(device)
                    obs_o = torch.Tensor(obs["obj"]).to(device)
                    obs_t = {"masks": obs_m, "obj": obs_o}
                else:
                    obs_t = torch.Tensor(obs).to(device)

            return episodic_returns


        rewards = evaluate_local(agent, make_env, 10,
                                 env_id=args.env_id,
                                 capture_video=args.capture_video,
                                 run_dir=writer_dir,
                                 device=device)

        wandb.summary["FinalReward_mean"] = float(np.mean(rewards))
        wandb.summary["FinalReward_median"] = float(np.median(rewards))
        wandb.summary["FinalReward_min"] = float(np.min(rewards))
        wandb.summary["FinalReward_max"] = float(np.max(rewards))
        wandb.log({"eval/RewardHist": wandb.Histogram(rewards)}, step=global_step)

        # Log video of agent's performance
        if args.capture_video:
            import os, glob

            video_dir = f"{writer_dir}/media/videos"
            videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
            if videos:
                # Log a couple of the most recent videos to the Media panel
                for v in videos[-2:]:
                    wandb.log({"video": wandb.Video(v, fps=30, format="mp4")}, step=global_step)
                # Force-upload all videos to the Files tab immediately
                wandb.save(os.path.join(video_dir, "*.mp4"), policy="now")

                # Also keep a versioned artifact with all videos
                va = wandb.Artifact(f"{args.exp_name}-videos", type="videos")
                for v in videos:
                    va.add_file(v)
                run.log_artifact(va, aliases=["latest"])

        wandb.finish()

    # Close environments and writer after training is complete
    envs.close()
    writer.close()
