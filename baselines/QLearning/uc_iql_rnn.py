import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import gymnax
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_vals


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions_i: Any
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def get_cf_greedy_actions(q_vals_i, q_vals_c, valid_actions):
        # mask unavailable actions
        unavail_actions = 1 - valid_actions
        q_vals_i = q_vals_i - (unavail_actions * 1e10)
        q_vals_c = q_vals_c - (unavail_actions * 1e10)

        # get action with max value for each set of q values
        act_i = jnp.argmax(q_vals_i, axis=-1, keepdims=True)
        act_c = jnp.argmax(q_vals_c, axis=-1, keepdims=True)

        # select greedy action as the action with the greater value across the two sets
        max_q_vals = jnp.maximum(q_vals_i, q_vals_c)
        greedy_actions = jnp.argmax(max_q_vals, axis=-1)

        return act_i, act_c, greedy_actions

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals_i, q_vals_c, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        act_i, act_c, cf_greedy_actions = get_cf_greedy_actions(q_vals_i, q_vals_c, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, cf_greedy_actions.shape)
            < eps,  # pick the actions that should be random
            random_actions,
            cf_greedy_actions,
        )
        return chosed_actions, act_i

    # uc exploration
    def uc_eps_greedy_exploration(rng, q_vals_i, q_vals_c, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        act_i, act_c, cf_greedy_actions = get_cf_greedy_actions(q_vals_i, q_vals_c, valid_actions)

        # choose random actions based regret decision criterion
        def get_cf_random_actions(rng, q_vals_i, q_vals_c, act_i, act_c):
            # mask unavailable actions
            unavail_actions = 1 - valid_actions
            q_vals_i = q_vals_i - (unavail_actions * 1e10)
            q_vals_c = q_vals_c - (unavail_actions * 1e10)

            # get corresponding value for each action
            q_act_i = jnp.take_along_axis(q_vals_i, act_i, axis=-1).squeeze(-1)
            q_act_c = jnp.take_along_axis(q_vals_c, act_c, axis=-1).squeeze(-1)

            # Assume initially both actions are equally weighted w [1,1].
            # diff > 0, w [1, 1 + diff]
            # diff <= 0, w [1 - diff, 1]
            diff = q_act_c - q_act_i
            w_i = jnp.ones_like(diff)
            w_c = jnp.ones_like(diff)
            w_c = jnp.where(diff > 0, 1 + diff, 1)
            w_i = jnp.where(diff <= 0, 1 - diff, 1)

            # normalize weights so that w[0] + w[1] = 1
            w_sum = w_i + w_c
            w_i /= w_sum
            w_c /= w_sum

            # choose act_i if random value is less than prob_i
            random_vals = jax.random.uniform(rng, shape=w_i.shape)
            selected_actions = jnp.where(random_vals < w_i, act_i.squeeze(-1), act_c.squeeze(-1))

            return selected_actions

        cf_random_actions = get_cf_random_actions(rng_a, q_vals_i, q_vals_c, act_i, act_c)

        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, cf_greedy_actions.shape)
            < eps,  # pick the actions that should be random
            cf_random_actions,
            cf_greedy_actions,
        )
        return chosen_actions, act_i

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"]
        )  # batched env for testing (has different batch size)

        # INIT I AND C NETWORKS AND OPTIMIZERS
        network_i = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        network_c = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        def create_agent(rng):
            # intention q network only uses observation as input
            init_x_i = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )

            # cf q network uses both observation and the intention network's output as input
            init_x_c = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size + wrapped_env.max_action_space)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )

            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)

            network_params_i = network_i.init(rng, init_hs, *init_x_i)
            network_params_c = network_c.init(rng, init_hs, *init_x_c)

            # log param count
            param_count_i = sum(x.size for x in jax.tree_util.tree_leaves(network_params_i))
            param_count_c = sum(x.size for x in jax.tree_util.tree_leaves(network_params_c))
            wandb.log({"agent_param_count": param_count_i+param_count_c})

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            # intention q network train state
            train_state_i = CustomTrainState.create(
                apply_fn=network_i.apply,
                params=network_params_i,
                target_network_params=network_params_i,
                tx=tx,
            )

            # cf q network train state
            train_state_c = CustomTrainState.create(
                apply_fn=network_c.apply,
                params=network_params_c,
                target_network_params=network_params_c,
                tx=tx,
            )

            return train_state_i, train_state_c

        rng, _rng = jax.random.split(rng)
        train_state_i, train_state_c = create_agent(rng)

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(
                jax.random.PRNGKey(0), 3
            )  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            actions_i = {
                agent: jnp.zeros((config["NUM_ENVS"], wrapped_env.max_action_space))
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions_i=actions_i,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree_map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state_i, train_state_c, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs_i, hs_c, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                # get q vals for intention q network
                new_hs_i, q_vals_i = jax.vmap(
                    network_i.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    train_state_i.params,
                    hs_i,
                    _obs,
                    _dones,
                )
                q_vals_i = q_vals_i.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

                # get q vals for frozen intention q network
                _, q_vals_fi = jax.vmap(
                    network_i.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    train_state_i.target_network_params,
                    hs_i,
                    _obs,
                    _dones,
                )
                q_vals_fi = q_vals_fi.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

                # get q vals for cf q network
                new_hs_c, q_vals_c = jax.vmap(
                    network_c.apply, in_axes=(None, 0, 0, 0)
                )(  # vmap across the agent dim
                    train_state_c.params,
                    hs_c,
                    jnp.concatenate([_obs, q_vals_fi[:, np.newaxis]], axis=-1), # condition on both observation and intention network q vals
                    _dones,
                )
                q_vals_c = q_vals_c.squeeze(
                    axis=1
                )

                # explore
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                eps = eps_scheduler(train_state_i.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                if config["EXP_MODE"] == "uc":
                    actions, actions_i = jax.vmap(uc_eps_greedy_exploration, in_axes=(0, 0, 0, None, 0))(
                        _rngs, q_vals_i, q_vals_c, eps, batchify(avail_actions)
                    )
                else:
                    actions, actions_i = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, 0, None, 0))(
                        _rngs, q_vals_i, q_vals_c, eps, batchify(avail_actions)
                    )
                actions = unbatchify(actions)
                q_vals_fi = unbatchify(q_vals_fi)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions_i=q_vals_fi,
                    actions=actions,
                    rewards=jax.tree_map(lambda x:config.get("REW_SCALE", 1)*x, rewards),
                    dones=dones,
                    avail_actions=avail_actions,
                )
                return (new_hs_i, new_hs_c, new_obs, dones, new_env_state, rng), (timestep, infos)

            # step the env (should be a complete rollout)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (init_hs, init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state_i = train_state_i.replace(
                timesteps=train_state_i.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            train_state_c = train_state_c.replace(
                timesteps=train_state_c.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state_i, train_state_c, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree_map(
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...)

                # preprocess network input
                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions_i = batchify(minibatch.actions_i)
                _actions = batchify(minibatch.actions)
                _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)

                # run target intention q network
                _, q_next_target_i = jax.vmap(network_i.apply, in_axes=(None, 0, 0, 0))(
                    train_state_i.target_network_params,
                    init_hs,
                    _obs,
                    _dones,
                )  # (num_agents, timesteps, batch_size, num_actions)

                # run target cf q network
                _, q_next_target_c = jax.vmap(network_c.apply, in_axes=(None, 0, 0, 0))(
                    train_state_c.target_network_params,
                    init_hs,
                    jnp.concatenate([_obs, _actions_i], axis=-1),
                    _dones,
                )  # (num_agents, timesteps, batch_size, num_actions)

                def _loss_fn_i(params_i):
                    # recompute q values for intention q network
                    _, q_vals_i = jax.vmap(network_i.apply, in_axes=(None, 0, 0, 0))(
                        params_i,
                        init_hs,
                        _obs,
                        _dones,
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    # get logits of the chosen actions for intention q network
                    chosen_action_q_vals_i = jnp.take_along_axis(
                        q_vals_i,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)
                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals_i = q_vals_i - (unavailable_actions * 1e10)

                    # get the q values of the next state for intention q network
                    q_next_i = jnp.take_along_axis(
                        q_next_target_i,
                        jnp.argmax(valid_q_vals_i, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)

                    # target y for intention q network
                    target_i = (
                        _rewards[:, :-1]
                        + (1 - _dones[:, :-1]) * config["GAMMA"] * q_next_i[:, 1:]
                    )

                    # loss for intention q network
                    chosen_action_q_vals_i = chosen_action_q_vals_i[:, :-1]
                    loss_i = jnp.mean(
                        (chosen_action_q_vals_i - jax.lax.stop_gradient(target_i)) ** 2
                    )

                    return loss_i, chosen_action_q_vals_i.mean()

                def _loss_fn_c(params_c):
                    # recompute q values for cf q network
                    _, q_vals_c = jax.vmap(network_c.apply, in_axes=(None, 0, 0, 0))(
                        params_c,
                        init_hs,
                        jnp.concatenate([_obs, _actions_i], axis=-1),
                        _dones,
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    # get logits of the chosen actions for cf q network
                    chosen_action_q_vals_c = jnp.take_along_axis(
                        q_vals_c,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)
                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals_c = q_vals_c - (unavailable_actions * 1e10)

                    # get the q values of the next state for cf q network
                    q_next_c = jnp.take_along_axis(
                        q_next_target_c,
                        jnp.argmax(valid_q_vals_c, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(
                        -1
                    )  # (num_agents, timesteps, batch_size,)

                    # target y for cf q network
                    target_c = (
                        _rewards[:, :-1]
                        + (1 - _dones[:, :-1]) * config["GAMMA"] * q_next_c[:, 1:]
                    )

                    # loss for cf q network
                    chosen_action_q_vals_c = chosen_action_q_vals_c[:, :-1]
                    loss_c = jnp.mean(
                        (chosen_action_q_vals_c - jax.lax.stop_gradient(target_c)) ** 2
                    )

                    return loss_c, chosen_action_q_vals_c.mean()


                # update intention q network
                (loss_i, qvals_i), grads_i = jax.value_and_grad(_loss_fn_i, has_aux=True)(
                    train_state_i.params
                )
                train_state_i = train_state_i.apply_gradients(grads=grads_i)
                train_state_i = train_state_i.replace(
                    grad_steps=train_state_i.grad_steps + 1,
                )

                # update cf q network
                (loss_c, qvals_c), grads_c = jax.value_and_grad(_loss_fn_c, has_aux=True)(
                    train_state_c.params
                )
                train_state_c = train_state_c.apply_gradients(grads=grads_c)
                train_state_c = train_state_c.replace(
                    grad_steps=train_state_c.grad_steps + 1,
                )

                return (train_state_i, train_state_c, rng), (loss_i, loss_c, qvals_i, qvals_c)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state_i.timesteps > config["LEARNING_STARTS"]
            )
            (train_state_i, train_state_c, rng), (loss_i, loss_c, qvals_i, qvals_c) = jax.lax.cond(
                is_learn_time,
                lambda train_state_i, train_state_c, rng: jax.lax.scan(
                    _learn_phase, (train_state_i, train_state_c, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state_i, train_state_c, rng: (
                    (train_state_i, train_state_c, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),  # do nothing
                train_state_i,
                train_state_c,
                _rng,
            )

            # update intention target network
            train_state_i = jax.lax.cond(
                train_state_i.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state_i: train_state_i.replace(
                    target_network_params=optax.incremental_update(
                        train_state_i.params,
                        train_state_i.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state_i: train_state_i,
                operand=train_state_i,
            )

            # update cf target network
            train_state_c = jax.lax.cond(
                train_state_c.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state_c: train_state_c.replace(
                    target_network_params=optax.incremental_update(
                        train_state_c.params,
                        train_state_c.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state_c: train_state_c,
                operand=train_state_c,
            )

            # UPDATE METRICS
            train_state_i = train_state_i.replace(n_updates=train_state_i.n_updates + 1)
            train_state_c = train_state_c.replace(n_updates=train_state_c.n_updates + 1)
            act_i = jnp.argmax(qvals_i, axis=-1)
            act_c = jnp.argmax(qvals_c, axis=-1)
            act_mismatch = act_i != act_c
            metrics = {
                "env_step": train_state_i.timesteps,
                "update_steps": train_state_i.n_updates,
                "grad_steps": train_state_i.grad_steps,
                "loss_i": loss_i.mean(),
                "loss_c": loss_c.mean(),
                "qvals_i": qvals_i.mean(),
                "qvals_c": qvals_c.mean(),
                "act_mismatch": act_mismatch.mean(),
            }
            metrics.update(jax.tree_map(lambda x: x.mean(), infos))
            if config.get("LOG_AGENTS_SEPARATELY", False):
                for i, a in enumerate(env.agents):
                    m = jax.tree_map(
                        lambda x: x[..., i].mean(),
                        infos,
                    )
                    m = {k + f"_{a}": v for k, v in m.items()}
                    metrics.update(m)

            # update the test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state_i.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state_i, train_state_c),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        metrics.update(
                            {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
                        )
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state_i, train_state_c, buffer_state, test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, train_state_i, train_state_c):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            params_i = train_state_i.params
            params_c = train_state_c.params
            def _greedy_env_step(step_state, unused):
                params_i, params_c, env_state, last_obs, last_dones, hstate_i, hstate_c, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                new_hstate_i, q_vals_i = jax.vmap(network_i.apply, in_axes=(None, 0, 0, 0))(
                    params_i,
                    hstate_i,
                    _obs,
                    _dones,
                )
                q_vals_i = q_vals_i.squeeze(axis=1)
                _, q_vals_fi = jax.vmap(network_i.apply, in_axes=(None, 0, 0, 0))(
                    train_state_i.target_network_params,
                    hstate_i,
                    _obs,
                    _dones,
                )
                q_vals_fi = q_vals_fi.squeeze(axis=1)
                new_hstate_c, q_vals_c = jax.vmap(network_c.apply, in_axes=(None, 0, 0, 0))(
                    params_c,
                    hstate_c,
                    jnp.concatenate([_obs, q_vals_fi[:, np.newaxis]], axis=-1),
                    _dones,
                )
                q_vals_c = q_vals_c.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                _, _, actions = get_cf_greedy_actions(q_vals_i, q_vals_c, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params_i, params_c, env_state, obs, dones, new_hstate_i, new_hstate_c, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                params_i,
                params_c,
                env_state,
                init_obs,
                init_dones,
                hstate,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            if config.get("LOG_AGENTS_SEPARATELY", False):
                metrics = {}
                for i, a in enumerate(env.agents):
                    m = jax.tree_map(
                        lambda x: jnp.nanmean(
                            jnp.where(
                                infos["returned_episode"][..., i],
                                x[..., i],
                                jnp.nan,
                            )
                        ),
                        infos,
                    )
                    m = {k + f"_{a}": v for k, v in m.items()}
                    metrics.update(m)
            else:
                metrics = jax.tree_map(
                    lambda x: jnp.nanmean(
                        jnp.where(
                            infos["returned_episode"],
                            x,
                            jnp.nan,
                        )
                    ),
                    infos,
                )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state_i, train_state_c)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_i, train_state_c, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init neeeds a scenario
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "mpe" in env_name.lower():
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "uc_iql_rnn")
    env, env_name= env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # save params
    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}  # merge the alg config with the main config
    env_name = default_config["ENV_NAME"]
    alg_name = default_config.get("ALG_NAME", "uc_iql_rnn")
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.005,
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)

    # force multiruns to finish correctly
    wandb.finish()


if __name__ == "__main__":
    main()
