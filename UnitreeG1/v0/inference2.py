import os

# Wayland + GLFW without libdecor => no title bar, window can't be moved.
# Run the viewer through XWayland (X11) instead. Must happen before the viewer
# is launched. Set USE_WAYLAND=1 to opt out.
if os.environ.get('XDG_SESSION_TYPE') == 'wayland' and os.environ.get('DISPLAY') \
        and not os.environ.get('USE_WAYLAND'):
    os.environ.pop('WAYLAND_DISPLAY', None)
    os.environ['XDG_SESSION_TYPE'] = 'x11'

import time
import sys
import threading
import select
import collections
import functools

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import mujoco.viewer

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp
from brax.training.acme import running_statistics

# ---------------------------------------------------------------------------
# Keyboard input
#
# WHY THE OLD VERSION FAILED: the MuJoCo viewer binds almost every letter key
# to a rendering toggle (W = wireframe, S = static bodies, A/D/Q/E = camera,
# frames, etc.). Those built-in handlers fire in addition to key_callback, so
# WASD changed the scene instead of (only) steering the robot.
#
# FIX: read the keys from the TERMINAL instead (stdin, cbreak mode). The viewer
# never sees them, so there are no conflicts. Keep the terminal focused while
# you drive; the viewer window is just for display.
# ---------------------------------------------------------------------------
LIN_SPEED = 1.0   # m/s   (training range: -1..1)
LAT_SPEED = 0.5   # m/s   (training range: -0.5..0.5)
YAW_SPEED = 1.0   # rad/s (training range: -1..1)

KEY_TO_CMD = {
    'w': (LIN_SPEED, 0.0, 0.0),
    's': (-LIN_SPEED, 0.0, 0.0),
    'a': (0.0, LAT_SPEED, 0.0),
    'd': (0.0, -LAT_SPEED, 0.0),
    'q': (0.0, 0.0, YAW_SPEED),
    'e': (0.0, 0.0, -YAW_SPEED),
    ' ': (0.0, 0.0, 0.0),
}

# Plain Python tuple (not a jax array) -> safe to share between threads.
current_cmd = (0.0, 0.0, 0.0)
stop_event = threading.Event()
reset_event = threading.Event()

GLFW_KEY_BACKSPACE = 259


def viewer_key_callback(keycode):
    """Backspace in the viewer window = reset the GPU simulation."""
    if keycode == GLFW_KEY_BACKSPACE:
        reset_event.set()


def keyboard_thread():
    """Reads single key presses from the terminal without needing Enter."""
    global current_cmd
    if not sys.stdin.isatty():
        print("stdin is not a TTY; keyboard control disabled.")
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                ch = sys.stdin.read(1).lower()
                if ch == 'r':
                    reset_event.set()
                    print("\rreset requested                    ", end="", flush=True)
                elif ch in KEY_TO_CMD:
                    current_cmd = KEY_TO_CMD[ch]
                    print(f"\rcommand [vx, vy, yaw] = {current_cmd}      ", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


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

    # 1. Build network with normalizer
    ppo_network = network_factory(
        eval_env.observation_size,
        eval_env.action_size,
        preprocess_observations_fn=running_statistics.normalize
    )

    # 2. Load checkpoint via Orbax
    checkpointer = ocp.PyTreeCheckpointer()
    raw_params = checkpointer.restore(checkpoint_path)

    normalizer_keys = raw_params[0].keys()
    DynamicRunningStats = collections.namedtuple('DynamicRunningStats', normalizer_keys)
    params = (
        DynamicRunningStats(**raw_params[0]),
        raw_params[1]
    )

    make_policy = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_policy(params)

    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # 3. Initialize batched state (batch size 1)
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    reset_rng = jax.random.split(reset_rng, 1)
    state = jit_reset(reset_rng)

    print("state.info keys:", list(state.info.keys()))

    # The env randomly RESAMPLES the joystick command every few seconds inside
    # step(), which would fight our keyboard command. Push that timer far into
    # the future so only our command is ever used.
    for k in ('steps_until_next_cmd', 'steps_since_last_resample'):
        if k in state.info:
            if k == 'steps_until_next_cmd':
                state.info[k] = jnp.full_like(state.info[k], 10**9)
            else:
                state.info[k] = jnp.zeros_like(state.info[k])

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Warm-up: compile policy + step BEFORE opening the viewer (first call takes
    # ~1 minute). The result is discarded, so the robot's state is untouched.
    print("Compiling policy and physics step (about a minute the first time)...")
    _rng, _act_rng = jax.random.split(rng)
    _action, _ = jit_inference_fn(state.obs, _act_rng)
    _warm = jit_step(state, _action)
    jax.block_until_ready(_warm.obs)
    del _warm
    print("Compiled. Launching viewer.")

    print("\n" + "=" * 44)
    print("CONTROLS (type in THIS TERMINAL, not the viewer):")
    print("W / S : Forward / Backward")
    print("A / D : Sidestep Left / Right")
    print("Q / E : Rotate Left / Right")
    print("SPACE : Stop (stand still)")
    print("R     : Reset robot (also Backspace in the viewer window)")
    print("=" * 44)
    print("Viewer: scroll / two-finger scroll = zoom, left-drag = rotate,")
    print("        right-drag = pan. (The viewer's own Reset button does not")
    print("        affect the GPU sim; use R or Backspace.)\n")

    global current_cmd
    kb = threading.Thread(target=keyboard_thread, daemon=True)
    kb.start()

    # Camera follows the robot so it never walks out of view.
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    if body_id < 0:
        body_id = 1

    # 4. Viewer
    try:
        with mujoco.viewer.launch_passive(
                mj_model, mj_data, key_callback=viewer_key_callback) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = body_id
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15

            while viewer.is_running():
                step_start = time.time()

                # Handle reset requests (terminal 'r' or Backspace in viewer)
                if reset_event.is_set():
                    reset_event.clear()
                    rng, reset_rng = jax.random.split(rng)
                    state = jit_reset(jax.random.split(reset_rng, 1))
                    current_cmd = (0.0, 0.0, 0.0)

                # Inject keyboard command; keep dtype/shape identical to the original.
                old_cmd = state.info['command']
                cmd = jnp.asarray([current_cmd], dtype=old_cmd.dtype)
                new_info = {**state.info, 'command': cmd}
                if 'steps_until_next_cmd' in new_info:
                    new_info['steps_until_next_cmd'] = jnp.full_like(
                        new_info['steps_until_next_cmd'], 10**9)
                state = state.replace(info=new_info)

                # Step GPU physics
                rng, act_rng = jax.random.split(rng)
                action, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, action)

                # Unbatch and sync to CPU viewer (MuJoCo Playground uses .data)
                unbatched_state = jax.tree_util.tree_map(lambda x: x[0], state.data)
                # NOTE: in current MJX, get_data() RETURNS a new MjData; to copy into
                # an existing one use get_data_into(result, model, mjx_data).
                mjx.get_data_into(mj_data, mj_model, unbatched_state)
                mujoco.mj_kinematics(mj_model, mj_data)  # refresh body poses for rendering
                viewer.sync()

                # Throttle to real time
                time_until_next_step = env.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        stop_event.set()
        kb.join(timeout=1.0)
        print()


if __name__ == '__main__':
    main()
