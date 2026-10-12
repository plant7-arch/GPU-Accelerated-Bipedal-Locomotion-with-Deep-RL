import os
# Optimizes XLA for faster training with Triton GEMM (recommended for modern GPUs)
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'

import jax
from datetime import datetime
from pathlib import Path
import orbax.checkpoint as ocp
# from orbax.checkpoint import utils as orbax_utils

# Brax and MuJoCo Playground Imports
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

def main():
    # 1. Select the Unitree G1 Environment
    env_name = 'G1JoystickFlatTerrain'
    
    # Load Environment & Configs
    print(f"Loading environment: {env_name}")
    # 2. Get Configs and Increase GPU Physics Buffers
    env_cfg = registry.get_default_config(env_name)
    
    # DeepMind's ConfigDict requires unlocking to modify defaults
    with env_cfg.unlocked():
        env_cfg.njmax = 300         # Fixes the 'nefc' constraint overflow
        env_cfg.naccdmax = 1000     # Fixes the 'CCD' collision overflow
        env_cfg.nconmax = 100       # Adds padding for contact points
        
    print(f"Loading environment: {env_name}")
    # Make sure to pass the updated config into the registry loader!
    env = registry.load(env_name, config=env_cfg)
    
    # 2. Get Default PPO Parameters for the G1
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_training_params = dict(ppo_params)
    
    # The network factory is passed differently in brax, so we pop it from the dict
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        
    # Domain Randomization (Friction, link masses, Torso CoM)
    randomizer = registry.get_domain_randomizer(env_name)
    
    # 3. Setup Checkpointing Configuration
    ckpt_path = Path("checkpoints").resolve() / env_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {ckpt_path}")
    
    ## Revert to PyTreeCheckpointer, which we know initialized correctly
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    
    def policy_params_fn(current_step, make_policy, params):
        """Callback to save the policy checkpoint at intervals"""
        path = (ckpt_path / f"step_{current_step}").as_posix() 
        
        # Just pass params directly as the second argument, no kwargs needed
        orbax_checkpointer.save(path, params, force=True)
        print(f"Saved checkpoint at step {current_step}")
        
    # 4. Train the Model using Brax PPO
    print("Starting PPO Training...")
    
    make_inference_fn, params, metrics = ppo.train(
        environment=env,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomizer,
        policy_params_fn=policy_params_fn,
        **ppo_training_params
    )
    
    print("Training Completed Successfully!")

if __name__ == "__main__":
    print(f"JAX running on: {jax.default_backend()}")
    main()
