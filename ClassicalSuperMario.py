import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, time, datetime, os
import argparse

import pennylane as qml
import matplotlib.pyplot as plt
# from pennylane import numpy as pnp # Not strictly needed for this QCNN but often useful

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--circuit-depth", type=int, default=2)
    parser.add_argument("--n-chips", type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--exploration-rate-decay', type=float, default=0.99999975)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--learn-step', type=int, default=3)
    parser.add_argument('--num-episodes', type=int, default=40000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log-index", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()
args=get_args()


def set_global_seeds(seed_value):
    """Sets global seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value) # For hash-based operations
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU (if used non-Lightning)
    # Potentially set for CuDNN
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Global seeds set to: {seed_value}")


RUN_ID = f"ClassicalMario_Run{args.log_index}"  # Keep this THE SAME for a given experiment run
CHECKPOINT_BASE_DIR = Path("SuperMarioCheckpoints") # Main folder for all runs
SAVE_DIR = CHECKPOINT_BASE_DIR / RUN_ID # Specific directory for this run
REPLAY_BUFFER_DIR = SAVE_DIR / "replay_buffer"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
REPLAY_BUFFER_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE_PATH = SAVE_DIR / "latest_checkpoint.chkpt" # Fixed name


# Initialize Super Mario environment
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

# Reset environment for the first state
env.reset()
# You can test the environment with a sample step:
# next_state, reward, done, trunc, info = env.step(action=0)
# print(f"{next_state.shape},\n {reward},\n {done},\n {info}")



class ClassicalCNN(nn.Module):
    def __init__(self, hidden_dim=8, depth=2):
        """
        Classical counterpart of the QCNN model using fully connected, convolutional, and pooling layers.

        Args:
            input_dim (int): Input dimension (784 for MNIST).
            hidden_dim (int): Dimension of the hidden embedding (matching n_qubits in QCNN).
            depth (int): Depth of the convolutional layers (matching QCNN circuit depth).
        """
        super().__init__()

        # Classical dimension reduction (embedding)
        # self.fc = nn.Linear(input_dim, hidden_dim)

        # Convolutional and pooling layers
        self.layers = nn.ModuleList()
        current_dim = hidden_dim
        for _ in range(depth):
            conv_pool_block = nn.Sequential(
                nn.Linear(current_dim, current_dim), 
                nn.ReLU(),
                nn.Linear(current_dim, current_dim // 2),
                nn.ReLU()
            )
            self.layers.append(conv_pool_block)
            current_dim = current_dim // 2

        # Final linear layer to get single logit output
        self.final_fc = nn.Linear(current_dim, 1)

    def forward(self, x):
        # # Classical dimension reduction
        # x = self.fc(x)

        # Convolutional and pooling-like layers
        for layer in self.layers:
            x = layer(x)

        # Final output (single logit)
        x = self.final_fc(x)

        return x.squeeze()


class ClassicalMarioNet(nn.Module):
    def __init__(self, input_dim_tuple, output_dim_actions, device,
                 n_qubits_per_chip=8, circuit_depth_per_chip=2, num_chips=1):
        super().__init__()
        
        C, H, W = input_dim_tuple
        self.qcnn_input_flattened_dim = C * H * W
        self.output_dim_actions = output_dim_actions

        self.n_qubits_for_qcnn = n_qubits_per_chip    # Qubits for each individual QCNN chip
        self.circuit_depth_for_qcnn = circuit_depth_per_chip
        self.num_chips = num_chips
        
        # self.input_dim_per_chip = self.qcnn_input_flattened_dim // self.num_chips
        # if self.qcnn_input_flattened_dim % self.num_chips != 0:
        #     raise ValueError("Flattened input dimension must be perfectly divisible by the number of chips.")
        # if self.input_dim_per_chip != self.n_qubits_for_qcnn: # Assuming direct feature-to-qubit mapping in QCNN's embedding
        #      print(f"Warning: input_dim_per_chip ({self.input_dim_per_chip}) "
        #            f"differs from n_qubits_per_chip ({self.n_qubits_for_qcnn}). "
        #            "Ensure QCNN's AngleEmbedding and internal FC layer are configured accordingly.")


        self.classical_hidden_dim = 256 # Size of hidden layer in the classical head
        self.classical_hidden_dim2 = 64 # Size of hidden layer in the classical head
        self.device = device

        self.online = self._build_complete_q_network()
        self.target = self._build_complete_q_network()
        
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def _build_complete_q_network(self):
        qcnn_chip_params = {
            'n_qubits': self.n_qubits_for_qcnn,
            'circuit_depth': self.circuit_depth_for_qcnn,
            # 'input_dim': self.input_dim_per_chip # Each QCNN chip is initialized to expect this many input features
        }

        classical_dimreduc_module = nn.Sequential(
            nn.Linear(self.qcnn_input_flattened_dim, self.classical_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.classical_hidden_dim, self.num_chips*self.n_qubits_for_qcnn)
        )
        
        # The classical head will receive an input vector of size num_chips (since each chip outputs 1 scalar)
        classical_head_input_dim = self.num_chips
        classical_head_module = nn.Sequential(
            nn.Linear(classical_head_input_dim, self.classical_hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.classical_hidden_dim2, self.output_dim_actions)
        )
        
        return FullQModel(qcnn_chip_params, self.num_chips, classical_dimreduc_module, classical_head_module, self.device)

    def forward(self, state_batch, model): # model is "online" or "target"
        return self.online(state_batch) if model == "online" else self.target(state_batch)


class FullQModel(nn.Module):
    def __init__(self, qcnn_chip_params, num_chips, dimreduc_module, head_module, device):
        super().__init__()
        self.flatten = nn.Flatten()
        self.num_chips = num_chips

        # Create a ModuleList of QCNN chips, each configured identically but with its own parameters
        self.qcnn_chips = nn.ModuleList([
            ClassicalCNN(hidden_dim=qcnn_chip_params['n_qubits'],
                 depth=qcnn_chip_params['circuit_depth'])
            for _ in range(self.num_chips)
        ])
        self.classical_dimreduc_part = dimreduc_module
        self.classical_head_part = head_module
        self.device = device

    def forward(self, x_input_frames): # x_input_frames shape: (batch_size, C, H, W)
        flattened_batch = self.flatten(x_input_frames).to(self.device) # Shape: (batch_size, 28224)
        reduced_batch = self.classical_dimreduc_part(flattened_batch)
        
        if reduced_batch.dtype != torch.float32:
            reduced_batch = reduced_batch.to(torch.float32)

        batch_size = reduced_batch.shape[0]
        
        outputs_from_chips_for_batch = [] # Will store tensors of shape (num_chips) for each batch item

        for i in range(batch_size):
            single_sample_flattened = reduced_batch[i] # Shape: (28224)
            
            # Split this single sample's 28224 features into num_chips chunks.
            # Each chunk will have 12 features.
            feature_chunks_for_sample = torch.chunk(single_sample_flattened, self.num_chips, dim=0)
            # feature_chunks_for_sample is a tuple of 'num_chips' tensors, each of shape (12,)

            current_sample_chip_outputs = [] # Scalar outputs from each chip for this one sample
            for chip_idx in range(self.num_chips):
                chip_input_features = feature_chunks_for_sample[chip_idx] # Shape: (12)
                
                # Each QCNN chip processes its dedicated 12-feature chunk
                # (Ensure QCNN.forward can take (12,) and returns a scalar tensor)
                chip_scalar_output = self.qcnn_chips[chip_idx](chip_input_features)
                current_sample_chip_outputs.append(chip_scalar_output)
            
            # For one sample, stack scalar outputs from all chips to form a vector of shape (num_chips)
            outputs_from_chips_for_batch.append(torch.stack(current_sample_chip_outputs))

        # Stack results for all samples in the batch: creates a tensor of shape (batch_size, num_chips)
        final_combined_output = torch.stack(outputs_from_chips_for_batch).to(self.device)
                                
        if final_combined_output.dtype != torch.float32:
            final_combined_output = final_combined_output.to(torch.float32)
        
        q_values = self.classical_head_part(final_combined_output)
        return q_values


    
class Mario:
    def __init__(self, state_dim, action_dim, save_dir, replay_buffer_path): # Added replay_buffer_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir # For general run artifacts if any, checkpoints go to specific path
        self.replay_buffer_path = replay_buffer_path

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = args.device
        
        # Assuming ClassicalMarioNet's __init__ is: (input_dim_tuple, output_dim_actions, device, ...)
        self.net = ClassicalMarioNet(self.state_dim, self.action_dim, self.device,
                                  args.n_qubits, args.circuit_depth, args.n_chips).to(self.device)

        self.exploration_rate = 1.0
        # self.exploration_rate_decay = 0.99999975
        self.exploration_rate_decay = args.exploration_rate_decay
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Initialize replay buffer - LazyMemmapStorage needs an existing directory
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, scratch_dir=str(self.replay_buffer_path), device=torch.device("cpu"))
        )

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025) # Ensure all net params are passed
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4
        self.learn_every = args.learn_step
        self.sync_every = 1e4
        self.save_every_episodes = 1 # Example: Save checkpoint every 50 episodes

    # act, cache, recall, td_estimate, td_target, update_Q_online, sync_Q_target methods remain largely the same
    # ... (ensure your existing methods are here) ...
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            # Ensure state is correctly preprocessed to a tensor
            if isinstance(state, tuple): state = state[0] # From env.reset() or FrameStack
            if not isinstance(state, torch.Tensor):
                 # Assuming state is LazyFrame or similar, convert to numpy then tensor
                state_np = np.array(state, dtype=np.float32)
                state = torch.tensor(state_np, device=self.device, dtype=torch.float32).unsqueeze(0)
            elif state.device != self.device or state.dtype != torch.float32: # If already tensor but wrong type/dev
                state = state.to(device=self.device, dtype=torch.float32).unsqueeze(0)
            else: # Already a correct tensor, just add batch dim
                state = state.unsqueeze(0)

            if state.ndim == 3: # If it's (C,H,W) -> (1,C,H,W)
                state = state.unsqueeze(0)


            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x): # Handle LazyFrame from FrameStack
            return x[0] if isinstance(x, tuple) else x

        # Convert LazyFrames to numpy arrays before making tensors
        state_np = np.array(first_if_tuple(state), dtype=np.float32)
        next_state_np = np.array(first_if_tuple(next_state), dtype=np.float32)

        # Store as CPU tensors in replay buffer for memory efficiency if GPU memory is tight
        state_t = torch.tensor(state_np, dtype=torch.float32)
        next_state_t = torch.tensor(next_state_np, dtype=torch.float32)
        action_t = torch.tensor([action], dtype=torch.int64)
        reward_t = torch.tensor([reward], dtype=torch.float32)
        done_t = torch.tensor([done], dtype=torch.bool) # Ensure boolean

        experience = TensorDict({
            "state": state_t, "next_state": next_state_t, "action": action_t,
            "reward": reward_t, "done": done_t
        }, batch_size=[]) # Empty batch_size for single experience
        self.memory.add(experience)


    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device) # Move to device for training
        # Ensure correct dtypes if not already handled by buffer or .to(device)
        state = batch.get("state").float()
        next_state = batch.get("next_state").float()
        action = batch.get("action").squeeze(-1)
        reward = batch.get("reward").squeeze(-1).float()
        done = batch.get("done").squeeze(-1).bool()
        return state, next_state, action, reward, done

    def td_estimate(self, state, action):
        # Assuming state is already (batch_size, C, H, W) and on correct device/dtype
        current_Q_values = self.net(state, model="online")
        # Action might be (batch_size), needs to be (batch_size, 1) for gather
        current_Q_selected = current_Q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        return current_Q_selected

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q_online = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q_online, axis=1)
        
        next_Q_target = self.net(next_state, model="target")
        best_next_q = next_Q_target.gather(1, best_action.unsqueeze(1)).squeeze(1)
        
        return (reward + (1 - done.float()) * self.gamma * best_next_q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        # In ClassicalMarioNet, online and target are attributes of self.net
        self.net.target.load_state_dict(self.net.online.state_dict())


    def save_checkpoint(self, episode, logger_state_dict, checkpoint_path):
        checkpoint = {
            'episode': episode,
            'curr_step': self.curr_step,
            'model_state_dict': self.net.state_dict(), # Saves ClassicalMarioNet (online & target)
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'logger_state': logger_state_dict,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path} (Episode {episode}, Step {self.curr_step})")

    def load_checkpoint(self, checkpoint_path):
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return 0, None # Start from episode 0, no logger state to load

        print(f"Loading checkpoint from {checkpoint_path}...")
        # map_location ensures CUDA tensors are loaded to self.device (CPU or CUDA)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.net.load_state_dict(checkpoint['model_state_dict'])
        # Target network is part of self.net.state_dict(), so it's also loaded.
        # self.sync_Q_target() # Optionally re-sync if there's any doubt, but load_state_dict should handle it.
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curr_step = checkpoint['curr_step']
        self.exploration_rate = checkpoint['exploration_rate']
        
        torch_rng_state = checkpoint['torch_rng_state']
        if torch_rng_state.device != torch.device('cpu'):
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        
        # Ensure optimizer state is moved to the correct device if necessary
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        completed_episode = checkpoint['episode']
        logger_state_to_load = checkpoint.get('logger_state', None)
        print(f"Checkpoint loaded. Resuming from Episode {completed_episode + 1} (Step {self.curr_step})")
        return completed_episode, logger_state_to_load

    def learn(self): # learn method remains the same, but no longer calls self.save()
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss) if td_est is not None else (0.0,loss)

        

class MetricLogger:
    def __init__(self, save_dir, resume_log_file_exists=False): # Renamed arg for clarity
        self.save_log_path = save_dir / "log.txt"
        
        log_mode = "a" if resume_log_file_exists else "w"
        
        with open(self.save_log_path, log_mode) as f:
            if log_mode == "w": # Write header only if creating a new log file
                f.write(
                    f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                    f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                    f"{'TimeDelta':>15}{'Time':>20}\n"
                )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # Initialize lists for metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode() # Initialize current episode trackers
        self.record_time = time.time()

    # log_step, log_episode, init_episode, record, plotting methods remain largely the same
    # ... (ensure your existing methods are here) ...
    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None and q is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        # Ensure lists are not empty before calculating mean, provide default if they are
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]) if self.ep_rewards else 0.0, 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]) if self.ep_lengths else 0.0, 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]) if self.ep_avg_losses else 0.0, 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]) if self.ep_avg_qs else 0.0, 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - Step {step} - Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_ep_reward} - Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        with open(self.save_log_path, "a") as f: # Always append after initial optional header
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
        self._plot_metrics() # Call plotting method

    def _plot_metrics(self): # Extracted plotting logic
        metrics_to_plot = {
            "Rewards": (self.moving_avg_ep_rewards, self.ep_rewards_plot),
            "Lengths": (self.moving_avg_ep_lengths, self.ep_lengths_plot),
            "Avg Losses": (self.moving_avg_ep_avg_losses, self.ep_avg_losses_plot),
            "Avg Qs": (self.moving_avg_ep_avg_qs, self.ep_avg_qs_plot)
        }
        for metric_name, (data, save_path) in metrics_to_plot.items():
            if data:
                plt.figure()
                plt.plot(data, label=f"Moving Avg {metric_name}")
                plt.title(f"Moving Average of Episode {metric_name}")
                plt.xlabel("Record Call (e.g., Episode)")
                plt.ylabel(metric_name)
                plt.legend()
                plt.savefig(save_path)
                plt.close()

    def get_state_dict(self):
        return {
            'ep_rewards': self.ep_rewards,
            'ep_lengths': self.ep_lengths,
            'ep_avg_losses': self.ep_avg_losses,
            'ep_avg_qs': self.ep_avg_qs,
            'moving_avg_ep_rewards': self.moving_avg_ep_rewards,
            'moving_avg_ep_lengths': self.moving_avg_ep_lengths,
            'moving_avg_ep_avg_losses': self.moving_avg_ep_avg_losses,
            'moving_avg_ep_avg_qs': self.moving_avg_ep_avg_qs,
            'record_time': self.record_time
        }

    def load_state_dict(self, state_dict):
        if state_dict:
            self.ep_rewards = state_dict.get('ep_rewards', [])
            self.ep_lengths = state_dict.get('ep_lengths', [])
            # ... load all other lists similarly ...
            self.moving_avg_ep_avg_qs = state_dict.get('moving_avg_ep_avg_qs', [])
            self.record_time = state_dict.get('record_time', time.time())
            print("MetricLogger state loaded.")

            

if __name__ == '__main__': # Important to put main execution under this
    set_global_seeds(args.seed)
    # --- Initialize Agent and Logger ---
    # Ensure env is defined before this if action_dim depends on it.
    # Example: temp_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    # temp_env = JoypadSpace(temp_env, [["right"], ["right", "A"]])
    # action_dim = temp_env.action_space.n
    # del temp_env
    # --- Initialize Mario and Logger ---
    action_dim = env.action_space.n # Make sure env is defined before this

    mario = Mario(state_dim=(4, 84, 84), action_dim=action_dim,
                  save_dir=SAVE_DIR, # Use the fixed SAVE_DIR
                  replay_buffer_path=REPLAY_BUFFER_DIR) # Use the fixed REPLAY_BUFFER_DIR

    start_episode = 0
    loaded_logger_state = None
    if CHECKPOINT_FILE_PATH.exists(): # Check for the checkpoint in the fixed path
        print(f"Found existing checkpoint: {CHECKPOINT_FILE_PATH}")
        completed_episode, loaded_logger_state = mario.load_checkpoint(CHECKPOINT_FILE_PATH)
        start_episode = completed_episode + 1 
    else:
        print(f"No checkpoint found at {CHECKPOINT_FILE_PATH}. Starting new training.")

    logger = MetricLogger(save_dir=SAVE_DIR, # Use the fixed SAVE_DIR
                          resume_log_file_exists=CHECKPOINT_FILE_PATH.exists())
    if loaded_logger_state:
        logger.load_state_dict(loaded_logger_state)

    # --- Training ---
    total_episodes = args.num_episodes # Your target

    print(f"Starting training from episode {start_episode} up to {total_episodes-1}.")
    print(f"Current step: {mario.curr_step}, Exploration rate: {mario.exploration_rate:.4f}")


    for e in range(start_episode, total_episodes):
        state_tuple = env.reset() # env should be defined globally or passed
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple

        while True:
            action = mario.act(state)
            
            next_state_tuple = env.step(action)
            if len(next_state_tuple) == 5: # gym new_step_api
                next_obs, reward, done_env, trunc, info = next_state_tuple
            else: # old gym api
                next_obs, reward, done_env, info = next_state_tuple
                trunc = False
            
            # The actual observation might be nested if FrameStack is used after new_step_api
            current_next_state = next_obs[0] if isinstance(next_obs, tuple) and not isinstance(next_obs[0], (int, float, bool)) else next_obs
            
            episode_done = done_env or trunc

            mario.cache(state, current_next_state, action, reward, episode_done)
            q_val, loss_val = mario.learn() # q_val can be None if still in burnin
            logger.log_step(reward, loss_val, q_val)
            
            state = current_next_state

            if episode_done or info.get("flag_get", False):
                break
        
        logger.log_episode()
        
        # Log overall episode metrics (e.g., every episode or every few)
        if e % 1 == 0 or e == total_episodes - 1 : # Log every episode for this example
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

        # Save checkpoint periodically
        if (e > 0 and e % mario.save_every_episodes == 0) or (e == total_episodes - 1):
            logger_state_to_save = logger.get_state_dict()
            mario.save_checkpoint(episode=e, logger_state_dict=logger_state_to_save, checkpoint_path=CHECKPOINT_FILE_PATH)
    
    print("Training finished.")



