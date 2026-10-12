import time
import collections
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import mujoco.viewer
import functools
import glfw

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp
from brax.training.acme import running_statistics

# Global command vector [vx, vy, yaw]
# Starts at 0.0 (standing still)
current_cmd = jnp.array([[0.0, 0.0, 0.0]])

def key_callback(keycode):
    """Updates the joystick command based on keyboard input."""
    global current_cmd
    
    # Velocity parameters
    lin_speed = 1.0
    lat_speed = 0.5
    yaw_speed = 1.0
    
    if keycode == glfw.KEY_W:
        current_cmd = jnp.array([[lin_speed, 0.0, 0.0]])
    elif keycode == glfw.KEY_S:
        current_cmd = jnp.array([[-lin_speed, 0.0, 0.0]])
    elif keycode == glfw.KEY_A:
        current_cmd = jnp.array([[0.0, lat_speed, 0.0]])
    elif keycode == glfw.KEY_D:
        current_cmd = jnp.array([[0.0, -lat_speed, 0.0]])
    elif keycode == glfw.KEY_Q:
        current_cmd = jnp.array([[0.0, 0.0, yaw_speed]])
    elif keycode == glfw.KEY_E:
        current_cmd = jnp.array([[0.0, 0.0, -yaw_speed]])
    elif keycode == glfw.KEY_SPACE: # Reset to standing still
        current_cmd = jnp.array([[0.0, 0.0, 0.0]])

def main():
    env_name = 'G1JoystickFlatTerrain'
    checkpoint_path = '/home/plant/Python/ML/Mujoco/mjx/mujoco_playground/UnitreeG1/v0/logs/G1JoystickFlatTerrain-20261003-234023/checkpoints/000202342400'
    
    env = registry.load(env_name, config_overrides={'impl': 'jax'})
    eval_env = wrapper.wrap_for_brax_training(env)
    
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.get('network_factory', {})
    )
    
    # 1. Build Network with Normalizer
    ppo_network = network_factory(
        eval_env.observation_size,
        eval_env.action_size,
        preprocess_observations_fn=running_statistics.normalize
    )
    
    # 2. Load Checkpoint via Orbax
    checkpointer = ocp.PyTreeCheckpointer()
    raw_params = checkpointer.restore(checkpoint_path)
    
    # Dynamically read the exact keys Brax saved (count, mean, and whatever the 3rd is)
    normalizer_keys = raw_params[0].keys()
    DynamicRunningStats = collections.namedtuple('DynamicRunningStats', normalizer_keys)
    
    # Cast the dictionary to our namedtuple using ** unpacking
    params = (
        DynamicRunningStats(**raw_params[0]),
        raw_params[1]
    )
    
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_policy(params)
    
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    
    # 3. Initialize Batched State
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    reset_rng = jax.random.split(reset_rng, 1) # Batch size of 1
    state = jit_reset(reset_rng)
    
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    print("\n" + "="*40)
    print("CONTROLS:")
    print("W / S : Move Forward / Backward")
    print("A / D : Sidestep Left / Right")
    print("Q / E : Rotate Left / Right")
    print("SPACE : Stop (Stand Still)")
    print("="*40 + "\n")

    # 4. Launch Viewer with Keyboard Listener
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Inject dynamic keyboard command
            new_info = {**state.info, 'command': current_cmd}
            state = state.replace(info=new_info)
            
            # Step GPU Physics
            rng, act_rng = jax.random.split(rng)
            action, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            
            # Unbatch and sync to CPU viewer
            unbatched_state = jax.tree_util.tree_map(lambda x: x[0], state.data)
            mjx.get_data(mj_model, mj_data, unbatched_state)
            viewer.sync()
            
            # Throttle playback to real-time
            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
