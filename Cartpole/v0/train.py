"""
train_cartpole.py
"""
import os
# Critical: Prevent JAX from pre-allocating all GPU memory so PyTorch (RSL-RL) can function
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
import jax
from rsl_rl.runners import OnPolicyRunner

# MuJoCo Playground imports
from mujoco_playground import registry
from mujoco_playground._src import wrapper_torch

def main():
    env_name = "CartpoleBalance"
    num_envs = 2048
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {env_name} via MuJoCo Playground (MJX)...")
    
    # 1. Load the default environment configuration
    env_cfg = registry.get_default_config(env_name)
    
    # 2. Instantiate the JAX-based MJX environment
    base_env = registry.load(env_name, config=env_cfg)
    
    # 3. Wrap for RSL-RL
    # This automatically vmaps the JAX env for batching and bridges JAX arrays to PyTorch tensors
    env = wrapper_torch.WrapperTorch(
        env=base_env,
        num_envs=num_envs,
        device=device
    )
    
    # 4. RSL-RL Configuration Dictionaries
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [64, 64],  # Compact MLP for Cartpole
            "critic_hidden_dims": [64, 64],
            "activation": "elu",
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "run_name": env_name,
            "experiment_name": "cartpole_rsl_rl",
            "max_iterations": 200,
            "num_steps_per_env": 64,
            "save_interval": 50,
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
            "resume_path": None,
        }
    }
    
    # 5. Initialize the RSL-RL Runner
    print(f"Initializing RSL-RL OnPolicyRunner on {device}...")
    log_dir = os.path.join("logs", train_cfg["runner"]["experiment_name"])
    
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=device
    )
    
    # 6. Execute Training Loop
    print("Starting training loop...")
    runner.learn(
        num_learning_iterations=train_cfg["runner"]["max_iterations"],
        init_at_random_ep_len=True
    )
    print(f"Training finished! Checkpoints saved to: {log_dir}")

if __name__ == "__main__":
    main()
