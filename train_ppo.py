# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy

from transformers import AutoImageProcessor, Dinov2Model, ConvNextV2Model

from einops import rearrange

# ---------------
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
from wrappers import MaxAndSkipEnv, JoypadSpace, VideoRecorder
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


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
    mps: bool = False
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
    total_timesteps: int = 100_000  # previously 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
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
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

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



def make_mario_env(env_id, idx, capture_video, run_name, n_frames=4):
    def thunk():
        env = gym_super_mario_bros.make(env_id, render_mode="rgb_array", apply_api_compatibility=True)

        # only capture video for first vector environment (idx 0)
        if capture_video and idx == 0:

            # takes in episode id and decides wether to record
            def should_record_episode(ep_id):
                return ep_id % 1 == 0
            
            env = VideoRecorder(
                env, 
                video_folder=f"train_videos/{run_name}", 
                should_record_episode=should_record_episode,
                fps=30
            )
        
        env = JoypadSpace(env, actions=SIMPLE_MOVEMENT)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #env = NoopResetEnv(env, noop_max=30)
        # TODO : implement No-op reset maybe 
        env = MaxAndSkipEnv(env, skip=4)
        #env = EpisodicLifeEnv(env)
        #if "FIRE" in env.unwrapped.get_action_meanings():
        #    env = FireResetEnv(env)
        #env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        #env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, n_frames)
        
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# -------------------------------
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

        # get CLS token
        features = outputs.last_hidden_state
        
        return features

class Dinov2_Encoder(nn.Module) : 

    def __init__(self) : 
        super().__init__()
        model_name="facebook/dinov2-small"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)

        self.embed_dim = self.model.config.hidden_size

        for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x) : 

        batch_size = x.shape[0]

        x = rearrange(x, 'b t h w c -> (b t) h w c')


        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        with torch.no_grad() : 

            outputs = self.model(**inputs)

        # get CLS token
        features = outputs.last_hidden_state[:, 0, :]  # [batch_size * n_frames, out_dim]

        assert self.embed_dim == features.shape[-1] # sanity check

        embeds = rearrange(features, '(b t) d -> b (t d)', b=batch_size)

        return embeds

'''
class PretrainedEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small"):
        super().__init__()
        
        print('trying to initialize model')
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        
        print('intialized models')

        # Freeze the pre-trained model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # x shape: [batch_size, frame_stack, h, w, 3]
        batch_size = x.size(0)
        frame_stack = x.size(1)
        
        print(f'first : {x.shape}')

        x = rearrange(x, 'b t h w c -> (b t) h w c')
        # Reshape to process all frames as a batch

        #x = x.reshape(-1, x.size(3), x.size(4), 3)  # [batch_size*frame_stack, h, w, 3]
        
        # Convert to RGB (if not already) and normalize
        #x = x / 255.0

        print(f'second : {x.shape}')
        
        # Prepare inputs for the DINOv2 model
        inputs = self.processor(images=x, return_tensors="pt")
        
        #print(inputs.keys())
        #print(input['pixel_vale'])

        #print(inputs.shape)

        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        
        print('moved to device')
        #exit(0)
        # Forward pass through DINOv2
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        print(f'outputs shape : {outputs.last_hidden_state.shape}')
        # Get CLS token features
        features = outputs.last_hidden_state[:, 0, :]  # [batch_size*frame_stack, feature_dim]
        
        print(f'third : {features.shape}')

        # Reshape back to include frame_stack dimension
        features = features.reshape(batch_size, frame_stack, -1)  # [batch_size, frame_stack, feature_dim]
        
        print(f'fourth : {features.shape}')
        # Flatten the frame_stack and feature dimensions
        features = features.reshape(batch_size, -1)  # [batch_size, frame_stack*feature_dim]
        
        return features

'''

class Agent(nn.Module):
    def __init__(self, n_frames, dim_action, encoder):
        super().__init__()

        '''
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        '''

        # Replace the CNN with a pre-trained encoder
        self.encoder = encoder()
        self.feature_dim = self.encoder.feature_dim
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, 128, kernel_size=1),       # Dimensionality reduction
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),         # Generate a single attention score per position
            nn.Sigmoid()                              # Normalize weights between 0-1
        )

        self.feature_projection = layer_init(nn.Linear(n_frames * self.feature_dim, 512))
        self.actor = layer_init(nn.Linear(512, dim_action), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_hidden(self,x) : 
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


    def get_value(self, x):
        hidden = self.get_hidden(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.get_hidden(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
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
    
    n_frames = 4
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_mario_env(cfg.env_id, i, cfg.capture_video, run_name, n_frames) for i in range(cfg.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(n_frames, 
                  dim_action=envs.single_action_space.n, 
                  encoder=ConvNextV2_Encoder
                  ).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=cfg.learning_rate, eps=1e-5, weight_decay=1e-2)

    # ALGO Logic: Storage setup
    obs = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    for iteration in range(1, cfg.num_iterations + 1):
        print(f'iteration {iteration}')
        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.num_steps):
            print(f'\tstep: {step}')

            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            next_done = torch.tensor(next_done, dtype=torch.float32).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)            

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

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
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()