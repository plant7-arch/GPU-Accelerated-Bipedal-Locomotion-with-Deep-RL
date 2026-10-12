"""Train a pick-and-place policy for the arm in pick_place_arm.xml (MJX + Brax PPO).

What is in here
---------------
* PickPlaceArm        : MJX environment (MuJoCo Playground `MjxEnv` API, same as your G1 env).
                        - asymmetric observations: noisy `state` for the actor, clean `privileged_state` for the critic
                        - sensor noise (per-step noise + per-episode calibration bias)
                        - random external force pulses (perturbations) on forearm / wrist / hand / cube
                        - randomised cube pose, yaw and goal every episode
* make_domain_randomizer : per-environment physics randomisation (friction, masses, damping, actuator gains, ...)
* train()             : Brax PPO training, saves Orbax checkpoints AND a plain pickle of the final parameters
* load_policy()       : rebuilds the inference function from the pickle (no Orbax / dict-vs-dataclass problems)

Usage
-----
    uv run python train_pick_place.py --smoke_test          # ~1 min, run this FIRST
    uv run python train_pick_place.py                       # full training (defaults: 4096 envs, 100M steps)
    uv run python train_pick_place.py --timesteps 200000000 --num_envs 8192

Notes
-----
* Everything uses impl='jax'. (Your warp install was broken in the G1 run, so the default avoids it.)
* Physics randomisation is drawn once per parallel environment (this is how Brax/Playground do it),
  while the task layout (cube, goal, yaw), sensor biases and pushes are re-drawn every episode.
"""
import argparse
import functools
import inspect
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env

XML_PATH = Path(__file__).resolve().parent / "pick_place_arm.xml"

NETWORK_KWARGS = dict(
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",              # noisy observations
    value_obs_key="privileged_state",    # clean observations + hidden physics info
)

# ----------------------------------------------------------------------------------------------
# Domain randomisation ranges (multiplicative scales unless stated otherwise)
# ----------------------------------------------------------------------------------------------
DEFAULT_DR = dict(
    cube_friction=(0.5, 1.5),      # absolute sliding friction of the cube
    finger_friction=(0.6, 1.5),    # absolute sliding friction of the fingers
    cube_mass=(0.5, 2.0),          # x nominal (50 g .. 200 g); inertia is scaled with it
    arm_mass=(0.85, 1.15),         # per body (links, hand, fingers)
    joint_damping=(0.7, 1.5),
    joint_frictionloss=(0.5, 1.5),
    joint_armature=(0.8, 1.3),
    kp_scale=(0.8, 1.2),           # per actuator position gain
    kv_scale=(0.8, 1.2),           # per actuator velocity gain
)


# ----------------------------------------------------------------------------------------------
# MJX version-tolerant helpers (the MJX API has been changing: impl=, nconmax/naconmax, njmax ...)
# ----------------------------------------------------------------------------------------------
def _call_filtered(fn, *args, **kwargs):
    """Call fn, dropping keyword arguments that this MJX version does not accept."""
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _put_model(mj_model, impl):
    return _call_filtered(mjx.put_model, mj_model, impl=impl)


def _make_data(mj_model, impl, cfg):
    extra = {}
    if impl == "warp":  # the jax implementation sizes contacts/constraints itself
        extra = dict(naconmax=cfg.naconmax, nconmax=cfg.naconmax, njmax=cfg.njmax)
    return _call_filtered(mjx.make_data, mj_model, impl=impl, **extra)


def _name2id(m, obj, name):
    i = mujoco.mj_name2id(m, obj, name)
    if i < 0:
        raise ValueError(f"'{name}' not found in {XML_PATH.name}")
    return i


def _v(x):
    """Scalar -> shape (1,) array so it can be concatenated."""
    return jnp.reshape(x, (1,))


def _yaw(R):
    """Heading of a frame's x axis in the world xy-plane."""
    return jnp.arctan2(R[1, 0], R[0, 0])


def _rel_yaw(cube_R, hand_R):
    """Yaw of the cube relative to the hand, wrapped to [-pi/4, pi/4) (the cube has 4-fold symmetry)."""
    rel = _yaw(cube_R) - _yaw(hand_R)
    return jnp.mod(rel + jnp.pi / 4, jnp.pi / 2) - jnp.pi / 4


# ----------------------------------------------------------------------------------------------
# Environment config
# ----------------------------------------------------------------------------------------------
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,              # 50 Hz policy
        sim_dt=0.004,              # 5 physics sub-steps per control step
        episode_length=400,        # 8 s
        impl="jax",
        naconmax=4096,             # only used by impl='warp'
        njmax=200,                 # only used by impl='warp'
        arm_action_scale=0.06,     # rad per step at |action| = 1 (incremental joint-target control)
        # ----- task layout (table top is z = 0; the cube is 4 cm) -----
        cube_x_range=[0.30, 0.50],
        cube_y_range=[-0.20, 0.20],
        min_goal_dist=0.12,
        init_qpos_noise=0.05,
        min_link_height=0.04,      # terminate if elbow / wrist go this close to the table
        # ----- sensor noise -----
        noise_config=config_dict.create(
            level=1.0,             # global multiplier, 0 disables all noise
            scales=config_dict.create(
                joint_pos=0.005,   # rad, uniform per-step noise
                joint_vel=0.05,    # rad/s
                gripper=0.001,     # m
                ee_pos=0.003,      # m
                cube_pos=0.005,    # m   (vision-like)
                cube_yaw=0.05,     # rad
                joint_bias=0.01,   # rad, constant per episode (encoder calibration error)
                cube_bias=0.005,   # m,   constant per episode (camera calibration error)
            ),
        ),
        # ----- perturbations: random force pulses applied to a random body -----
        push_config=config_dict.create(
            enable=True,
            interval_range=[1.0, 3.0],     # s between pushes
            duration_range=[0.1, 0.4],     # s per push
            magnitude_range=[0.3, 1.0],    # fraction of the body's max force
            bodies=["link3", "link5", "hand", "cube"],
            max_force=[15.0, 10.0, 8.0, 1.5],   # N
        ),
        # ----- reward -----
        reward_config=config_dict.create(
            lift_height=0.04,      # cube must rise this far above the table (while held) to count as "picked"
            grasp_dist=0.035,      # cube centre within this of the grip site => "near"
            closed_q=0.026,        # finger joint < this => gripper "closed" (cube is 0.016 when squeezed)
            open_q=0.03,           # finger joint > this => gripper "open"
            goal_tol=0.03,
            scales=config_dict.create(
                reach=1.0, orient=0.3, align=0.5, grasp=1.0, lift=2.0, goal=3.0, place=5.0,
                action_rate=-0.02, joint_vel=-0.001, joint_limit=-2.0,
            ),
        ),
    )


_REWARD_KEYS = ("reach", "orient", "align", "grasp", "lift", "goal", "place",
                "action_rate", "joint_vel", "joint_limit")
_METRIC_KEYS = tuple(f"reward/{k}" for k in _REWARD_KEYS) + ("success", "lifted", "d_goal", "d_reach")


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------
class PickPlaceArm(mjx_env.MjxEnv):
    """Pick the cube up and place it on the green goal marker.

    Action (7): 6 x joint-target increments, 1 x gripper opening (-1 = closed ... +1 = open).
    """

    def __init__(self, config=None, config_overrides=None):
        super().__init__(config if config is not None else default_config(), config_overrides)
        cfg = self._config
        m = mujoco.MjModel.from_xml_path(str(XML_PATH))
        m.opt.timestep = cfg.sim_dt
        self._mj_model = m
        # NOTE: step()/reset() must always read self._mjx_model (the domain-randomisation wrapper swaps it).
        self._mjx_model = _put_model(m, cfg.impl)
        self._init_data = _make_data(m, cfg.impl, cfg)
        self._n_substeps = int(round(cfg.ctrl_dt / cfg.sim_dt))
        self._cache_ids()

    # ---- required MjxEnv properties --------------------------------------------------------------
    @property
    def xml_path(self) -> str:
        return str(XML_PATH)

    @property
    def action_size(self) -> int:
        return 7

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ---- one-time lookups ------------------------------------------------------------------------
    def _cache_ids(self):
        m, cfg = self._mj_model, self._config
        J, B, G, S, K = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM,
                         mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_KEY)
        if m.nu != 8:
            raise ValueError(f"expected 8 actuators (6 arm + 2 fingers), got {m.nu}")

        arm_j = [_name2id(m, J, f"joint{i}") for i in range(1, 7)]
        self._arm_qadr = np.array([m.jnt_qposadr[j] for j in arm_j])
        self._arm_dadr = np.array([m.jnt_dofadr[j] for j in arm_j])
        self._arm_lo = m.jnt_range[arm_j, 0].astype(np.float32)
        self._arm_hi = m.jnt_range[arm_j, 1].astype(np.float32)

        fing_j = [_name2id(m, J, n) for n in ("finger_l_joint", "finger_r_joint")]
        self._fing_qadr = np.array([m.jnt_qposadr[j] for j in fing_j])
        self._grip_max = float(m.jnt_range[fing_j[0], 1])

        cube_j = _name2id(m, J, "cube_free")
        self._cube_qadr = int(m.jnt_qposadr[cube_j])
        self._cube_dadr = int(m.jnt_dofadr[cube_j])
        self._cube_body = _name2id(m, B, "cube")
        self._cube_geom = _name2id(m, G, "cube_geom")
        self._cube_half = float(m.geom_size[self._cube_geom, 0])
        self._hand_body = _name2id(m, B, "hand")
        self._ee_site = _name2id(m, S, "grip_site")

        # for domain randomisation
        self._finger_geoms = np.array([_name2id(m, G, "finger_l_geom"), _name2id(m, G, "finger_r_geom")])
        self._arm_bodies = np.array([_name2id(m, B, n) for n in
                                     ("link1", "link2", "link3", "link4", "link5", "hand", "finger_l", "finger_r")])
        # bodies that must stay above the table (arm links have no collision geometry)
        self._low_bodies = np.array([_name2id(m, B, n) for n in ("link3", "link4", "link5")])

        kid = _name2id(m, K, "home")
        self._home_qpos = np.array(m.key_qpos[kid], dtype=np.float32)

        # perturbations (control-step units)
        pc = cfg.push_config
        to_steps = lambda s: max(1, int(round(s / cfg.ctrl_dt)))  # noqa: E731
        self._push_enabled = bool(pc.enable)
        self._push_body_ids = jnp.asarray([_name2id(m, B, n) for n in pc.bodies], dtype=jnp.int32)
        self._push_max = jnp.asarray(pc.max_force, dtype=jnp.float32)
        self._push_int = (to_steps(pc.interval_range[0]), to_steps(pc.interval_range[1]))
        self._push_dur = (to_steps(pc.duration_range[0]), to_steps(pc.duration_range[1]))
        self._reward_scales = cfg.reward_config.scales.to_dict()

    def dr_indices(self) -> dict:
        """Indices needed by the domain randomiser."""
        return dict(cube_body=self._cube_body, cube_geom=self._cube_geom, finger_geoms=self._finger_geoms,
                    arm_bodies=self._arm_bodies, arm_dofs=self._arm_dadr)

    # ---- episode sampling (used by reset() AND at the start of an episode inside step()) ---------
    def _sample_episode(self, rng):
        cfg = self._config
        sc = cfg.noise_config.scales
        lvl = cfg.noise_config.level
        k_cube, k_yaw, k_goal, k_q, k_jb, k_cb, k_pt = jax.random.split(rng, 7)

        lo = jnp.array([cfg.cube_x_range[0], cfg.cube_y_range[0]])
        hi = jnp.array([cfg.cube_x_range[1], cfg.cube_y_range[1]])
        cube_xy = jax.random.uniform(k_cube, (2,), minval=lo, maxval=hi)
        yaw = jax.random.uniform(k_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
        # goal: first of 8 candidates that is far enough from the cube
        cand = jax.random.uniform(k_goal, (8, 2), minval=lo, maxval=hi)
        valid = jnp.linalg.norm(cand - cube_xy, axis=-1) > cfg.min_goal_dist
        goal_xy = cand[jnp.argmax(valid)]

        qpos = jnp.asarray(self._home_qpos, dtype=jnp.float32)
        q_noise = jax.random.uniform(k_q, (6,), minval=-cfg.init_qpos_noise, maxval=cfg.init_qpos_noise)
        q_arm = jnp.clip(qpos[self._arm_qadr] + q_noise, self._arm_lo, self._arm_hi)
        qpos = qpos.at[self._arm_qadr].set(q_arm)
        qpos = qpos.at[self._fing_qadr].set(self._grip_max)  # gripper starts open
        cube_qpos = jnp.array([cube_xy[0], cube_xy[1], self._cube_half + 1e-3,
                               jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
        qpos = qpos.at[self._cube_qadr:self._cube_qadr + 7].set(cube_qpos)
        qvel = jnp.zeros(self._mj_model.nv)

        ctrl = jnp.concatenate([q_arm, jnp.full((2,), self._grip_max)])
        target = jnp.concatenate([q_arm, jnp.full((1,), self._grip_max)])
        goal = jnp.array([goal_xy[0], goal_xy[1], self._cube_half])
        mocap = jnp.array([[goal_xy[0], goal_xy[1], 0.002]])

        info = dict(
            goal=goal,
            target=target,
            last_act=jnp.zeros(7),
            lifted=jnp.zeros(()),
            joint_bias=lvl * sc.joint_bias * jax.random.uniform(k_jb, (6,), minval=-1.0, maxval=1.0),
            cube_bias=lvl * sc.cube_bias * jax.random.uniform(k_cb, (3,), minval=-1.0, maxval=1.0),
            push_timer=jax.random.randint(k_pt, (), self._push_int[0], self._push_int[1] + 1),
            push_left=jnp.zeros((), jnp.int32),
            push_force=jnp.zeros(3),
            push_body=jnp.zeros((), jnp.int32),
        )
        return qpos, qvel, ctrl, mocap, info

    # ---- reset -----------------------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_ep, k_obs = jax.random.split(rng, 3)
        qpos, qvel, ctrl, mocap, ep_info = self._sample_episode(k_ep)
        data = self._init_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl, mocap_pos=mocap)
        data = mjx.forward(self._mjx_model, data)

        info = dict(rng=rng, t=jnp.zeros((), jnp.int32), rerand=jnp.zeros((), jnp.int32), **ep_info)
        obs = self._get_obs(data, info, k_obs)
        metrics = {k: jnp.zeros(()) for k in _METRIC_KEYS}
        return mjx_env.State(data=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()),
                             metrics=metrics, info=info)

    # ---- step ------------------------------------------------------------------------------------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        rc = cfg.reward_config
        data = state.data
        info = dict(state.info)  # keep the wrapper's own keys untouched
        rng, k_ep, k_push, k_noise = jax.random.split(info["rng"], 4)

        # 0) New episode inside step(): the Playground auto-reset wrapper may restore the *first* reset state
        #    of this environment, which would give every env a fixed cube/goal forever. Re-draw the layout here.
        #    (If the wrapper does a full reset instead, info['rerand'] is already 0 and nothing happens.)
        qpos_n, qvel_n, ctrl_n, mocap_n, ep_n = self._sample_episode(k_ep)
        rr = info["rerand"] > 0
        data = data.replace(
            qpos=jnp.where(rr, qpos_n, data.qpos),
            qvel=jnp.where(rr, qvel_n, data.qvel),
            ctrl=jnp.where(rr, ctrl_n, data.ctrl),
            mocap_pos=jnp.where(rr, mocap_n, data.mocap_pos),
        )
        for k, v in ep_n.items():
            info[k] = jnp.where(rr, v, info[k])

        # 1) action -> position targets (incremental for the arm, absolute for the gripper)
        action = jnp.clip(action, -1.0, 1.0)
        arm_t = jnp.clip(info["target"][:6] + action[:6] * cfg.arm_action_scale, self._arm_lo, self._arm_hi)
        grip_t = (action[6] + 1.0) * 0.5 * self._grip_max
        target = jnp.concatenate([arm_t, _v(grip_t)])
        ctrl = jnp.concatenate([arm_t, jnp.stack([grip_t, grip_t])])

        # 2) perturbation: random force pulses
        data, info = self._apply_push(data, info, k_push)

        # 3) physics
        data = self._physics(data, ctrl)

        # 4) measurements
        q = data.qpos[self._arm_qadr]
        qd = data.qvel[self._arm_dadr]
        q_f = data.qpos[self._fing_qadr].mean()
        ee = data.site_xpos[self._ee_site]
        cube = data.xpos[self._cube_body]
        hand_R = data.xmat[self._hand_body]
        cube_R = data.xmat[self._cube_body]
        cube_v = data.qvel[self._cube_dadr:self._cube_dadr + 3]
        rel_yaw = _rel_yaw(cube_R, hand_R)
        goal = info["goal"]

        # 5) reward
        d_reach = jnp.linalg.norm(ee - cube)
        near = d_reach < rc.grasp_dist
        closed = q_f < rc.closed_q
        holding = (near & closed).astype(jnp.float32)
        picked_now = (cube[2] > self._cube_half + rc.lift_height) & (holding > 0.5)
        lifted = jnp.maximum(info["lifted"], picked_now.astype(jnp.float32))  # sticky: cube was really lifted

        d_goal = jnp.linalg.norm(cube - goal)
        r_reach = 1.0 - jnp.tanh(5.0 * d_reach)
        placed = (lifted * (d_goal < rc.goal_tol) * (jnp.linalg.norm(cube_v) < 0.05) *
                  (q_f > rc.open_q)).astype(jnp.float32)
        lim_viol = (jnp.maximum(self._arm_lo + 0.05 - q, 0.0) + jnp.maximum(q - (self._arm_hi - 0.05), 0.0))

        terms = dict(
            reach=r_reach,
            orient=jnp.clip(-hand_R[2, 2], 0.0, 1.0),                       # tool axis pointing down
            align=r_reach * (1.0 - jnp.tanh(3.0 * jnp.abs(rel_yaw))),        # fingers aligned with cube faces
            grasp=holding,
            lift=(1.0 - lifted) * holding * jnp.clip((cube[2] - self._cube_half) / rc.lift_height, 0.0, 1.0),
            goal=lifted * (1.0 - jnp.tanh(5.0 * d_goal)),                    # only counts after a real pick
            place=placed,
            action_rate=jnp.sum(jnp.square(action - info["last_act"])),
            joint_vel=jnp.sum(jnp.square(qd)),
            joint_limit=jnp.sum(lim_viol),
        )
        rewards = {k: terms[k] * self._reward_scales[k] for k in _REWARD_KEYS}
        reward = jnp.nan_to_num(sum(rewards.values()) * cfg.ctrl_dt)

        # 6) termination (time-limit truncation is handled by the Brax EpisodeWrapper)
        fell = cube[2] < -0.02
        out_of_bounds = jnp.any(jnp.abs(cube[:2] - jnp.array([0.4, 0.0])) > jnp.array([0.6, 0.5]))
        arm_low = jnp.any(data.xpos[self._low_bodies, 2] < cfg.min_link_height)
        bad = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
        terminated = fell | out_of_bounds | arm_low | bad
        done = terminated.astype(jnp.float32)

        # 7) book-keeping + observation
        t_new = info["t"] + 1
        episode_over = terminated | (t_new >= cfg.episode_length)
        info["t"] = jnp.where(episode_over, 0, t_new)
        info["rerand"] = episode_over.astype(jnp.int32)
        info["lifted"] = lifted
        info["last_act"] = action
        info["target"] = target
        info["rng"] = rng
        obs = jax.tree_util.tree_map(jnp.nan_to_num, self._get_obs(data, info, k_noise))

        metrics = {f"reward/{k}": v for k, v in rewards.items()}
        metrics.update(success=placed, lifted=lifted, d_goal=d_goal, d_reach=d_reach)
        return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    # ---- physics / perturbations -----------------------------------------------------------------
    def _physics(self, data, ctrl):
        model = self._mjx_model  # (possibly randomised) per-env model

        def sub_step(d, _):
            return mjx.step(model, d.replace(ctrl=ctrl)), None

        data, _ = jax.lax.scan(sub_step, data, (), length=self._n_substeps)
        return data

    def _apply_push(self, data, info, rng):
        pc = self._config.push_config
        k_body, k_dir, k_mag, k_dur, k_int = jax.random.split(rng, 5)
        body_idx = jax.random.randint(k_body, (), 0, self._push_body_ids.shape[0])
        direction = jax.random.normal(k_dir, (3,))
        direction = direction / (jnp.linalg.norm(direction) + 1e-6)
        mag = jax.random.uniform(k_mag, (), minval=pc.magnitude_range[0], maxval=pc.magnitude_range[1])
        mag = mag * self._push_max[body_idx]
        dur = jax.random.randint(k_dur, (), self._push_dur[0], self._push_dur[1] + 1)
        interval = jax.random.randint(k_int, (), self._push_int[0], self._push_int[1] + 1)

        timer = info["push_timer"] - 1
        start = (timer <= 0) & self._push_enabled
        info["push_left"] = jnp.where(start, dur, jnp.maximum(info["push_left"] - 1, 0))
        info["push_force"] = jnp.where(start, direction * mag, info["push_force"])
        info["push_body"] = jnp.where(start, body_idx, info["push_body"])
        info["push_timer"] = jnp.where(start, interval + dur, timer)

        applied = jnp.where(info["push_left"] > 0, info["push_force"], 0.0)
        xfrc = jnp.zeros((self._mj_model.nbody, 6)).at[self._push_body_ids[info["push_body"]], :3].set(applied)
        return data.replace(xfrc_applied=xfrc), info

    # ---- observations ----------------------------------------------------------------------------
    def _get_obs(self, data, info, rng):
        cfg = self._config
        sc, lvl = cfg.noise_config.scales, cfg.noise_config.level
        q = data.qpos[self._arm_qadr]
        qd = data.qvel[self._arm_dadr]
        grip = data.qpos[self._fing_qadr].mean()
        ee = data.site_xpos[self._ee_site]
        cube = data.xpos[self._cube_body]
        rel_yaw = _rel_yaw(data.xmat[self._cube_body], data.xmat[self._hand_body])
        goal, lifted, last_act = info["goal"], info["lifted"], info["last_act"]

        # ----- noisy "sensor" readings for the actor -----
        ks = jax.random.split(rng, 6)
        u = lambda k, n: jax.random.uniform(k, (n,), minval=-1.0, maxval=1.0)  # noqa: E731
        q_n = q + info["joint_bias"] + lvl * sc.joint_pos * u(ks[0], 6)
        qd_n = qd + lvl * sc.joint_vel * u(ks[1], 6)
        grip_n = grip + lvl * sc.gripper * u(ks[2], 1)[0]
        ee_n = ee + lvl * sc.ee_pos * u(ks[3], 3)
        cube_n = cube + info["cube_bias"] + lvl * sc.cube_pos * u(ks[4], 3)
        yaw_n = rel_yaw + lvl * sc.cube_yaw * u(ks[5], 1)[0]

        def pack(q_, qd_, g_, ee_, cube_, yaw_):
            return jnp.concatenate([q_, qd_, _v(g_), ee_, cube_, goal, cube_ - ee_, goal - cube_,
                                    _v(yaw_), _v(lifted), last_act])

        state = pack(q_n, qd_n, grip_n, ee_n, cube_n, yaw_n)

        # ----- clean + hidden information for the critic -----
        applied_push = jnp.where(info["push_left"] > 0, info["push_force"], 0.0)
        privileged = jnp.concatenate([
            pack(q, qd, grip, ee, cube, rel_yaw),
            data.qvel[self._cube_dadr:self._cube_dadr + 6],                 # cube lin + ang velocity
            applied_push,                                                    # current external push
            _v(self._mjx_model.body_mass[self._cube_body]),                  # randomised cube mass
            _v(self._mjx_model.geom_friction[self._cube_geom, 0]),           # randomised cube friction
        ])
        return {"state": state, "privileged_state": privileged}


# ----------------------------------------------------------------------------------------------
# Domain randomisation: returns (batched_model, in_axes) exactly like Playground's randomizers.
# Brax calls it as  fn(mjx_model, rng=<(num_envs, 2) keys>)  and passes it to the vmap wrapper.
# ----------------------------------------------------------------------------------------------
def make_domain_randomizer(env: PickPlaceArm, num_envs: int, dr: dict | None = None):
    dr = dr or DEFAULT_DR
    ix = env.dr_indices()
    nbody, nv, nu = env.mj_model.nbody, env.mj_model.nv, env.mj_model.nu
    n_arm_bodies, n_arm_dofs = len(ix["arm_bodies"]), len(ix["arm_dofs"])
    names = ("geom_friction", "body_mass", "body_inertia", "dof_damping", "dof_frictionloss",
             "dof_armature", "actuator_gainprm", "actuator_biasprm")

    def domain_randomize(model: mjx.Model, rng=None):
        if rng is None:
            rng = jax.random.split(jax.random.PRNGKey(0), num_envs)

        @jax.vmap
        def rand(key):
            k = jax.random.split(key, 10)
            U = lambda kk, shape, r: jax.random.uniform(kk, shape, minval=r[0], maxval=r[1])  # noqa: E731

            geom_friction = (model.geom_friction
                             .at[ix["cube_geom"], 0].set(U(k[0], (), dr["cube_friction"]))
                             .at[ix["finger_geoms"], 0].set(U(k[1], (), dr["finger_friction"])))

            body_scale = jnp.ones(nbody)
            body_scale = body_scale.at[ix["cube_body"]].set(U(k[2], (), dr["cube_mass"]))
            body_scale = body_scale.at[ix["arm_bodies"]].set(U(k[3], (n_arm_bodies,), dr["arm_mass"]))
            body_mass = model.body_mass * body_scale
            body_inertia = model.body_inertia * body_scale[:, None]   # inertia scales with mass

            def dof_scale(kk, rng_):
                return jnp.ones(nv).at[ix["arm_dofs"]].set(U(kk, (n_arm_dofs,), rng_))
            dof_damping = model.dof_damping * dof_scale(k[4], dr["joint_damping"])
            dof_frictionloss = model.dof_frictionloss * dof_scale(k[5], dr["joint_frictionloss"])
            dof_armature = model.dof_armature * dof_scale(k[6], dr["joint_armature"])

            s_kp = U(k[7], (nu,), dr["kp_scale"])
            s_kv = U(k[8], (nu,), dr["kv_scale"])
            gainprm = model.actuator_gainprm.at[:, 0].set(model.actuator_gainprm[:, 0] * s_kp)
            biasprm = (model.actuator_biasprm
                       .at[:, 1].set(model.actuator_biasprm[:, 1] * s_kp)   # -kp follows gain
                       .at[:, 2].set(model.actuator_biasprm[:, 2] * s_kv))  # -kv
            return (geom_friction, body_mass, body_inertia, dof_damping, dof_frictionloss,
                    dof_armature, gainprm, biasprm)

        values = rand(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({n: 0 for n in names})
        model = model.tree_replace(dict(zip(names, values)))
        return model, in_axes

    return domain_randomize


# ----------------------------------------------------------------------------------------------
# Building blocks shared by smoke test and training
# ----------------------------------------------------------------------------------------------
def build_env(args) -> PickPlaceArm:
    cfg = default_config()
    cfg.impl = args.impl
    if args.no_noise:
        cfg.noise_config.level = 0.0
    if args.no_push:
        cfg.push_config.enable = False
    return PickPlaceArm(cfg)


def smoke_test(args):
    """Build everything, run a few randomised vectorised steps, and check shapes / finiteness."""
    n = 16
    env = build_env(args)  # a *separate* instance: the DR wrapper mutates the env it wraps
    print(f"obs sizes: {env.observation_size}  action size: {env.action_size}  "
          f"dt={env.dt}  substeps={env._n_substeps}")
    rand_fn = make_domain_randomizer(env, n) if not args.no_dr else None
    randomization_fn = (functools.partial(rand_fn, rng=jax.random.split(jax.random.PRNGKey(1), n))
                        if rand_fn else None)
    wrapped = wrapper.wrap_for_brax_training(
        env, episode_length=env._config.episode_length, action_repeat=1, randomization_fn=randomization_fn)

    reset, step = jax.jit(wrapped.reset), jax.jit(wrapped.step)
    state = reset(jax.random.split(jax.random.PRNGKey(0), n))
    key = jax.random.PRNGKey(2)
    total = 0.0
    t0 = time.time()
    for i in range(60):
        key, k = jax.random.split(key)
        state = step(state, jax.random.uniform(k, (n, 7), minval=-1.0, maxval=1.0))
        total += float(state.reward.mean())
    jax.block_until_ready(state.obs)
    for name, o in state.obs.items():
        assert bool(jnp.all(jnp.isfinite(o))), f"non-finite values in obs['{name}']"
        print(f"  obs['{name}'] shape {o.shape}")
    assert bool(jnp.all(jnp.isfinite(state.reward)))
    print(f"  mean reward/step with random actions: {total / 60:.5f}")
    print("  cube z:", np.round(np.asarray(state.data.xpos[:, env._cube_body, 2])[:6], 3))
    print("  keys in info:", sorted(state.info.keys()))
    print(f"SMOKE TEST PASSED ({time.time() - t0:.1f}s incl. compilation)")


def train(args):
    env, eval_env = build_env(args), build_env(args)  # two instances (the DR wrapper mutates its env)
    cfg = env._config

    run_dir = Path(args.out_dir).resolve() / f"PickPlaceArm-{time.strftime('%Y%m%d-%H%M%S')}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(XML_PATH, run_dir / XML_PATH.name)
    print(f"Run directory: {run_dir}")

    num_minibatches = 8
    assert args.num_envs % num_minibatches == 0, "--num_envs must be a multiple of 8"
    ppo_kwargs = dict(
        num_timesteps=args.timesteps,
        num_evals=20,
        episode_length=cfg.episode_length,   # must match the env config (the env tracks episodes itself)
        action_repeat=1,
        num_envs=args.num_envs,
        batch_size=args.num_envs // num_minibatches,   # => batch_size * num_minibatches == num_envs
        num_minibatches=num_minibatches,
        unroll_length=20,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=0.005,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        reward_scaling=1.0,
        normalize_observations=True,
        num_resets_per_eval=1,
        seed=args.seed,
    )
    (run_dir / "config.json").write_text(json.dumps(dict(
        ppo=ppo_kwargs, network=NETWORK_KWARGS, env=cfg.to_dict(),
        domain_randomization=None if args.no_dr else DEFAULT_DR), indent=2, default=str))

    randomization_fn = None if args.no_dr else make_domain_randomizer(env, args.num_envs)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **NETWORK_KWARGS)

    t_start = time.time()

    def progress(step, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        s = float(metrics.get("eval/episode_success", float("nan")))
        print(f"[{(time.time() - t_start) / 60:7.1f} min] step {step:>12,d} | eval reward {r:8.3f} "
              f"| placed-steps/episode {s:7.2f}", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        randomization_fn=randomization_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        save_checkpoint_path=str(ckpt_dir),
        **ppo_kwargs,
    )

    # Plain pickle of (normalizer_state, policy_params, value_params): reload with load_policy().
    params_path = run_dir / "policy_params.pkl"
    model.save_params(str(params_path), params)
    print(f"\nTraining finished in {(time.time() - t_start) / 60:.1f} min")
    print(f"  final params : {params_path}")
    print(f"  checkpoints  : {ckpt_dir}")
    return params_path


def load_policy(params_path, env: PickPlaceArm | None = None, deterministic: bool = True):
    """Rebuild the jit-able inference function from policy_params.pkl.

    policy = load_policy('runs/.../policy_params.pkl')
    action, _ = jax.jit(policy)(obs_dict, jax.random.PRNGKey(0))     # obs_dict: {'state': (B,37), ...}
    """
    env = env or PickPlaceArm()
    params = model.load_params(str(params_path))
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **NETWORK_KWARGS)
    return ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=100_000_000)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--impl", default="jax", choices=["jax", "warp"])
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--no_dr", action="store_true", help="disable physics domain randomisation")
    p.add_argument("--no_noise", action="store_true", help="disable observation noise")
    p.add_argument("--no_push", action="store_true", help="disable random perturbation forces")
    p.add_argument("--smoke_test", action="store_true", help="quick end-to-end check, no training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    smoke_test(args) if args.smoke_test else train(args)
