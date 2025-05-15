# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
import gym
import torch
import wandb
import tyro
import yaml
import gym_super_mario_bros

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from pathlib import Path
from wrappers import MaxAndSkipEnv, JoypadSpace, VideoRecorder
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from transformers import AutoImageProcessor, Dinov2Model, ConvNextV2Model
from einops import rearrange

class Reward_Mode(Enum):
    EXTRINSIC = "extrinsic"
    INTRINSIC = "intrinsic"
    WEIGHTED = "weighted"

@dataclass
class Config:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    mps: bool = True
    """if toggled, mps will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SuperMarioBros-1-1-v0"
    """the id of the environment"""
    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    agent_learning_rate: float = 2.5e-4
    """the learning rate of the agent optimizer"""
    num_envs: int = 8
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
    agent_max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # ICM module parameters
    reward_mode: Reward_Mode = Reward_Mode.INTRINSIC
    """whether to use extrinsic only, intrinsic only, or weighted combination"""
    intrinsic_reward_scale: float = 0.2
    """scaling of intrinsic reward"""
    dim_obs_embed: int = 512
    """dimension of the observation embeddings"""
    icm_learning_rate: float = 2.5e-4
    """the learning rate of the icm module"""
    inverse_loss_weight: float = 1.0
    """weight balancing forward vs inverse losses"""
    forward_loss_weight: float = 1.0
    """weight balancing forward vs inverse losses"""
    icm_max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    checkpoint_interval: int = 10
    """save checkpoint every N iterations"""
    resume_from: Optional[str] = None
    """path to checkpoint to resume from"""

    config_file: Optional[str] = None
    """optional path to YAML config file"""

    def __post_init__(self):
        """Load values from config file if provided."""
        if self.config_file:
            self._load_from_config()
    
    def _load_from_config(self):
        """Load values from YAML config file."""
        
        with open(self.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Only override fields that are present in the config file
        # and that are part of our dataclass
        for key, value in config_data.items():
            if hasattr(self, key) :
                if key != "config_file":
                    setattr(self, key, value)
            else:
                raise ValueError(f"Unrecognized key in config file: '{key}'")
        
        print(f"Loaded configuration from {self.config_file}")



def make_mario_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym_super_mario_bros.make(env_id, render_mode="rgb_array", apply_api_compatibility=True)

        # only capture video for first vector environment (idx 0)
        if capture_video and idx == 0:

            # takes in episode id and decides wether to record
            def should_record_episode(ep_id):
                return ep_id % 1 == 0
            
            env = VideoRecorder(
                env, 
                video_folder=f"videos/{run_name}", 
                should_record_episode=should_record_episode,
                fps=30
            )
        
        env = JoypadSpace(env, actions=SIMPLE_MOVEMENT)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = NoopResetEnv(env, noop_max=30)
        # No-op reset maybe 
        env = MaxAndSkipEnv(env, skip=4)
        #env = EpisodicLifeEnv(env)
        #if "FIRE" in env.unwrapped.get_action_meanings():
        #    env = FireResetEnv(env)
        #env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        #env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Create checkpoint directory with function to save models
def save_checkpoint(run_name, agent, icm=None, agent_optimizer=None, icm_optimizer=None, 
                   iteration=0, global_step=0, iteration_max_x_pos=0, cfg=None):
    """Save training state to checkpoint file."""
    # Create checkpoint directory based on run_name (similar to runs and videos)
    save_dir = Path(f"checkpoints/{run_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint dictionary with all relevant state
    checkpoint = {
        "agent_state": agent.state_dict(),
        "agent_optimizer": agent_optimizer.state_dict() if agent_optimizer else None,
        "iteration": iteration,
        "global_step": global_step,
        "iteration_max_x_pos": iteration_max_x_pos,
        "cfg": vars(cfg) if cfg else None
    }
    
    # Add ICM state if available
    if icm is not None:
        checkpoint["icm_state"] = icm.state_dict()
    if icm_optimizer is not None:
        checkpoint["icm_optimizer"] = icm_optimizer.state_dict()
    
    # Save checkpoint
    checkpoint_path = save_dir / f"checkpoint_{iteration}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest checkpoint (for easy resuming)
    latest_path = save_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")
    return checkpoint_path

# Function to load checkpoint and resume training
def load_checkpoint(load_path, agent, icm=None, agent_optimizer=None, icm_optimizer=None):
    """Load training state from checkpoint file."""
    if not os.path.exists(load_path):
        print(f"Checkpoint not found at {load_path}")
        return None, 0, 0, 0
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # Load agent state
    agent.load_state_dict(checkpoint["agent_state"])
    if agent_optimizer and "agent_optimizer" in checkpoint:
        agent_optimizer.load_state_dict(checkpoint["agent_optimizer"])
    
    # Load ICM state if available
    if icm is not None and "icm_state" in checkpoint:
        icm.load_state_dict(checkpoint["icm_state"])
    if icm_optimizer is not None and "icm_optimizer" in checkpoint:
        icm_optimizer.load_state_dict(checkpoint["icm_optimizer"])
    
    # Get training state
    iteration = checkpoint.get("iteration", 0) 
    global_step = checkpoint.get("global_step", 0)
    iteration_max_x_pos = checkpoint.get("iteration_max_x_pos", 0)
    
    print(f"Loaded checkpoint from {load_path} (iteration {iteration}, global step {global_step})")
    return checkpoint, iteration, global_step, iteration_max_x_pos

# -------------------------------------------------------------


class Dinov2_Encoder(nn.Module) : 

    def __init__(self) : 
        super().__init__()
        model_name="facebook/dinov2-small"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)

        self.feature_dim = self.model.config.hidden_size

        for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x) : 

        batch_size = x.shape[0]

        x = rearrange(x, 'b t h w c -> (b t) h w c')


        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        with torch.no_grad() : 

            outputs = self.model(**inputs)

        features = outputs.last_hidden_state[:, 1:, :]  # [batch_size * n_frames, 256, embed_dim]

        assert self.feature_dim == features.shape[-1] # sanity check
        assert features.shape[-2] == 256 # sanity check

        embeds = rearrange(features, '(b t) (h w) d -> (b t) d h w', b=batch_size, h = 16)

        return embeds

class ConvNextV2_Encoder(nn.Module) : 

    def __init__(self) : 
        super().__init__()
        model_name="facebook/convnextv2-tiny-1k-224"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ConvNextV2Model.from_pretrained(model_name)

        self.feature_dim = 768

        for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x) : 

        batch_size = x.shape[0]

        x = rearrange(x, 'b t h w c -> (b t) h w c')

        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        with torch.no_grad() : 

            outputs = self.model(**inputs)

        features = outputs.last_hidden_state

        # (b t) d h w
        
        return features


class Agent(nn.Module):
    def __init__(self, dim_action_space, n_frames = 4):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(n_frames * 3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, dim_action_space), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):

        x = rearrange(x, 'b t h w c -> b (t c) h w')

        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):

        x = rearrange(x, 'b t h w c -> b (t c) h w')

        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


# idea : icm returns -- pred_next_action_logits ; next obs embed and pred next obs embed
class ICM(nn.Module): 

    def __init__(self, dim_action_space, 
                 dim_obs_embed = 512,
                 n_frames = 4,
                 encoder = ConvNextV2_Encoder): 

        super().__init__()
        #self.icm_weight = intrinsic_reward_weight
        #self.forward_loss_weight = forward_loss_weight

        self.encoder = encoder()
        self.feature_dim = self.encoder.feature_dim
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, 128, kernel_size=1),       # Dimensionality reduction
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),         # Generate a single attention score per position
            nn.Sigmoid()                              # Normalize weights between 0-1
        )
        self.feature_projection = layer_init(nn.Linear(n_frames * self.feature_dim, 512))

        # take obs embed + next obs embed and guess action taken
        self.inverse_model = nn.Sequential(
            nn.Linear(dim_obs_embed * 2, 512), 
            nn.ReLU(), 
            nn.Linear(512, dim_action_space)
        )

        # take feature + action and guess next feature 
        self.forward_model = nn.Sequential(
            nn.Linear(dim_obs_embed + dim_action_space, 512), 
            nn.ReLU(), 
            nn.Linear(512, dim_obs_embed)
        )

    def get_embed(self, x) : 
        # has shape b x t x h x w x c
        batch_size = x.shape[0]

        features = self.encoder(x)
        # (b t) d h w

        attention_weights = self.spatial_attention(features)
        # (b t) 1 h w

        attended_features = features * attention_weights
        # (b t) d h w

        global_features = attended_features.sum(dim=[-1, -2]) 
        # (b t) d

        global_features = rearrange(global_features, '(b t) d -> b (t d)', b=batch_size)
        # b (t d)

        hidden = self.feature_projection(global_features)

        return hidden

    def forward(self, curr_obs, next_obs, action_one_hot, rollout_mode = False): 

        curr_obs_embed = self.get_embed(curr_obs)
        next_obs_embed = self.get_embed(next_obs)

        pred_action_logits = None
        if not rollout_mode : 
            # predict action with inverse model
            pred_action_logits = self.inverse_model(torch.cat([curr_obs_embed, next_obs_embed], dim = -1))

        # predict next state with forward model
        pred_next_obs_embed = self.forward_model(torch.cat([curr_obs_embed, action_one_hot], dim = -1))

        return pred_action_logits, pred_next_obs_embed, next_obs_embed


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.reward_mode = Reward_Mode(cfg.reward_mode)  # ugly, but quick fix
    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    if cfg.track:
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=vars(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = "cpu"
    if cfg.cuda and torch.cuda.is_available() : 
        device = "cuda"
    elif cfg.mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() : 
        device = "mps"

    print(f"Using device: {device}")
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_mario_env(cfg.env_id, i, cfg.capture_video, run_name) for i in range(cfg.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs.single_action_space.n).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=cfg.agent_learning_rate, eps=1e-5, weight_decay=1e-2)


    if cfg.reward_mode != Reward_Mode.EXTRINSIC: 

        # initialize ICM module

        icm = ICM(envs.single_action_space.n,
                  dim_obs_embed= cfg.dim_obs_embed,
                  encoder = ConvNextV2_Encoder
                  ).to(device)
        
        icm_optimizer = optim.AdamW(icm.parameters(), lr=cfg.icm_learning_rate, eps=1e-5, weight_decay=1e-2)

    # ------------------------------- initialization done ---------------------------
    
    # --- memory allocation of vars -------------------------

    obs = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device)
    # step, env, dim_obs
    actions = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape).to(device)
    # step, env, dim_action
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    # step, env
    total_rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    # step, env
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    # step, env -> stores when envs finish
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    # step, env
    episode_max_x_pos = np.zeros(cfg.num_envs)

    if cfg.reward_mode == Reward_Mode.EXTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED: 

        extrinsic_rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        episode_extrinsic_rewards = np.zeros(cfg.num_envs)

    if cfg.reward_mode == Reward_Mode.INTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED : 

        intrinsic_rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        episode_intrinsic_rewards = np.zeros(cfg.num_envs)

        actions_onehot = torch.zeros((cfg.num_envs, envs.single_action_space.n)).to(device)

        next_obs_buffer = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device) 
    
    # Episode statistics (per environment)
    episode_total_rewards = np.zeros(cfg.num_envs)  
    episode_lengths = np.zeros(cfg.num_envs, dtype=int)

    # ----------------- Start the game ----------------------
    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    # resume from checkpoint
    start_iteration = 1
    if cfg.resume_from:
        checkpoint, loaded_iteration, loaded_global_step, loaded_max_x_pos = load_checkpoint(
            cfg.resume_from, agent, icm, optimizer, icm_optimizer
        )
        if loaded_iteration > 0:  # Successfully loaded
            start_iteration = loaded_iteration + 1
            global_step = loaded_global_step
            iteration_max_x_pos = loaded_max_x_pos
            print(f"Resuming from iteration {start_iteration}, global step {global_step}")

    for iteration in range(start_iteration, cfg.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.agent_learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # max x-pos across all environments
        iteration_max_x_pos = 0

        print(f"\nSTART: Iteration {iteration}")

        total_rewards.zero_()

        print('\nCollecting rollouts...')
        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # -------------------------------------------

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # Track x-position for each environment
            if "x_pos" in infos:
                current_x_pos = infos["x_pos"]
                # Update episode max x-position
                episode_max_x_pos = np.maximum(episode_max_x_pos, current_x_pos)
                # Update global max x-position
                iteration_max_x_pos = max(iteration_max_x_pos, current_x_pos.max())

            # Log global max position periodically
            if global_step % 1000 == 0:
                writer.add_scalar("charts/global_max_x_pos", iteration_max_x_pos, global_step)

            next_done = np.logical_or(terminations, truncations)
            next_done = torch.tensor(next_done, dtype=torch.float32).to(device)

            # Log data for extrinsic rewards
            if cfg.reward_mode == Reward_Mode.EXTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED: 
                
                # Update statistics
                extrinsic_rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
                episode_extrinsic_rewards += extrinsic_rewards[step].cpu().numpy()

                total_rewards[step] += extrinsic_rewards[step]


            if cfg.reward_mode == Reward_Mode.INTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED : 

                # store obs in buffer
                next_obs_buffer[step] = torch.Tensor(next_obs).to(device)

                # convert actions to one-hot
                actions_onehot.zero_()
                actions_onehot.scatter_(1, action.unsqueeze(1), 1.0)

                # Calculate intrinsic reward using ICM
                with torch.no_grad():
                    _, pred_next_obs_embed, next_obs_embed = icm(obs[step], 
                                                                 torch.tensor(next_obs).to(device), 
                                                                 actions_onehot, 
                                                                 rollout_mode = True)

                    # intrinsic reward is L2 dist between pred and true next obs embed 
                    # we zero out intrinsic reward for states right before reset
                    intrinsic_rewards[step] = (1.0 - next_done) * torch.mean(torch.square(pred_next_obs_embed - next_obs_embed), dim=-1)

                episode_intrinsic_rewards += intrinsic_rewards[step].cpu().numpy()
                total_rewards[step] += cfg.intrinsic_reward_scale * intrinsic_rewards[step]
                    
            episode_total_rewards += total_rewards[step].cpu().numpy()
            episode_lengths +=1
            
            # these are will be used in the next loop
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            #next_done = torch.tensor(next_done, dtype=torch.float32).to(device)

            if "final_info" in infos:
                
                # if at least one env has ended
                if not all(v is None for v in infos["final_info"]) : 
                    print(f"\n------------------- Global Step : {global_step} -------------------")

                    for env_idx, info in enumerate(infos["final_info"]):

                        if info and "episode" in info:  # environment at env_idx has episode ended

                            print(f"\tEnvironment {env_idx} finished with episode length {episode_lengths[env_idx]}")

                            writer.add_scalar("charts/episode_max_x_pos", episode_max_x_pos[env_idx], global_step)
                            print(f"\t\tMax x-position: {episode_max_x_pos[env_idx]}")

                            episode_max_x_pos[env_idx] = 0

                            if cfg.reward_mode == Reward_Mode.EXTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED: 
                                writer.add_scalar("charts/extrinsic_return", episode_extrinsic_rewards[env_idx], global_step)

                                print(f"\t\tExtrinsic return: {episode_extrinsic_rewards[env_idx]:.2f}")

                            if cfg.reward_mode == Reward_Mode.INTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED: 
                                writer.add_scalar("charts/intrinsic_return", episode_intrinsic_rewards[env_idx], global_step)

                                print(f"\t\tIntrinsic return: {episode_intrinsic_rewards[env_idx]:.2f}")
                            
                            writer.add_scalar("charts/total_return", episode_total_rewards[env_idx], global_step)
                            print(f"\t\tTotal return: {episode_total_rewards[env_idx]:.2f}")

                            episode_total_rewards[env_idx] = 0                   
                            
                            episode_lengths[env_idx] = 0

                            if cfg.reward_mode == Reward_Mode.EXTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED: 
                                episode_extrinsic_rewards[env_idx] = 0

                            if cfg.reward_mode == Reward_Mode.INTRINSIC or cfg.reward_mode == Reward_Mode.WEIGHTED: 
                                episode_intrinsic_rewards[env_idx] = 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(total_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = total_rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if cfg.reward_mode != Reward_Mode.EXTRINSIC : 

            b_next_obs = next_obs_buffer.reshape((-1,) + envs.single_observation_space.shape)
            b_dones = dones.reshape(-1)
            
            # For the last step's next observation, use the latest next_obs
            b_next_obs[-cfg.num_envs:] = next_obs
            
            # Create a mask to exclude transitions that cross episode boundaries
            # If done[t] is True, then the transition from t to t+1 crosses an episode boundary
            valid_transitions = ~b_dones.bool()
            
            # The last transitions of each environment need special handling
            # They're valid if the corresponding environment isn't done
            valid_transitions[-cfg.num_envs:] = ~next_done.bool()

            # Convert actions to one-hot
            b_actions_onehot = torch.zeros(b_actions.shape[0], envs.single_action_space.n).to(device)
            b_actions_onehot.scatter_(1, b_actions.long().unsqueeze(1), 1.0)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []

        inverse_loss = None
        forward_loss = None
        icm_loss = None

        print('\nUpdating models...')
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]


                # ------------------- Update Agent --------------------------

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.agent_max_grad_norm)
                optimizer.step()

                #----------- ICM UPDATE  -------------------

                if cfg.reward_mode != Reward_Mode.EXTRINSIC : 

                    
                    # Get batch data
                    mb_obs = b_obs[mb_inds]
                    mb_next_obs = b_next_obs[mb_inds]
                    mb_actions = b_actions.long()[mb_inds]
                    mb_actions_onehot = b_actions_onehot[mb_inds]
                    mb_valid = valid_transitions[mb_inds]

                    # Skip if no valid transitions in this minibatch
                    if not mb_valid.any():
                        continue

                    # Only use valid transitions for ICM learning
                    if not mb_valid.all():
                        mb_obs = mb_obs[mb_valid]
                        mb_next_obs = mb_next_obs[mb_valid]
                        mb_actions = mb_actions[mb_valid]
                        mb_actions_onehot = mb_actions_onehot[mb_valid]

                    pred_action_logits, pred_next_obs_embed, next_obs_embed = icm(
                        mb_obs, 
                        mb_next_obs, 
                        mb_actions_onehot,
                        rollout_mode=False
                    )

                    # Inverse model loss (action prediction)
                    inverse_loss = torch.nn.functional.cross_entropy(pred_action_logits, mb_actions.squeeze(-1))
                    
                    # Forward model loss (next state prediction)
                    forward_loss = torch.mean(torch.square(pred_next_obs_embed - next_obs_embed))

                    icm_loss = cfg.inverse_loss_weight * inverse_loss + cfg.forward_loss_weight * forward_loss

                    # Update ICM
                    icm_optimizer.zero_grad()
                    icm_loss.backward()
                    nn.utils.clip_grad_norm_(icm.parameters(), cfg.icm_max_grad_norm)
                    icm_optimizer.step()
                    


            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/agent_learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if cfg.reward_mode != Reward_Mode.EXTRINSIC and inverse_loss is not None:
            writer.add_scalar("losses/icm_inverse_loss", inverse_loss.item(), global_step)
            print(f"\tInverse Loss: {inverse_loss.item():.2f}. Scaled: {cfg.inverse_loss_weight*inverse_loss.item():.2f}")
            writer.add_scalar("losses/icm_forward_loss", forward_loss.item(), global_step)
            print(f"\tForward Loss: {forward_loss.item():.2f}. Scaled: {cfg.forward_loss_weight*forward_loss.item():.2f}")
            writer.add_scalar("losses/icm_total_loss", icm_loss.item(), global_step)
            print(f"\tTotal ICM Loss: {icm_loss.item():.2f}")
            writer.add_scalar("charts/mean_intrinsic_reward", intrinsic_rewards.mean().item(), global_step)

        print(f"\nIteration {iteration} over. Steps-per-second (SPS):", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        print(f"Iteration max x-pos: {iteration_max_x_pos}")
        writer.add_scalar("charts/iteration_max_x_pos", iteration_max_x_pos, iteration)   
        
        if cfg.checkpoint_interval is not None: 
            if (iteration - 1) % cfg.checkpoint_interval == 0 or iteration == cfg.num_iterations:
                print('Checkpoint saved.')
                save_checkpoint(
                    run_name, agent, icm, optimizer, icm_optimizer, 
                    iteration, global_step, iteration_max_x_pos, cfg
                )
    
    if cfg.checkpoint_interval is not None: 
        save_checkpoint(
            run_name, agent, icm, optimizer, icm_optimizer, 
            cfg.num_iterations, global_step, iteration_max_x_pos, cfg
        )
    
    envs.close()
    writer.close()