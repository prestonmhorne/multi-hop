# main.py

import numpy as np
import torch
import argparse
import config
from circuit_env import CircuitEnv
from dqn_agent import DQNAgent
from baseline_agent import BaselineAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='dqn',
                        choices=['baseline', 'dqn'])
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES)
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CircuitEnv()

    if args.agent == 'dqn':
        agent = DQNAgent(
            action_dim=config.NUM_RELAYS,
            state_dim=env.observation_space.shape[0]
        )

    elif args.agent == 'baseline':
        agent = BaselineAgent(
            action_dim=config.NUM_RELAYS,
            state_dim=env.observation_space.shape[0]
        )


    print(f"Training {agent.__class__.__name__}...")
    print(f"Episodes: {args.episodes}")
    print(f"Relays: {config.NUM_RELAYS}")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        terminated = False
        episode_reward = 0
        steps = 0

        while not terminated:
            action_mask = env.get_action_mask()
            
            if args.agent == 'baseline':
                relay_info = env.get_relay_info()
                action = agent.policy(obs, action_mask, relay_info)

            else:
                action = agent.policy(obs, action_mask, env.get_relay_info())

            next_obs, reward, terminated, _, _ = env.step(action)
            if args.agent == 'dqn':
                relay_info = env.get_relay_info()
                next_action_mask = env.get_action_mask()
                agent.update(obs, action, reward, next_obs, terminated, relay_info, relay_info, next_action_mask)

            obs = next_obs
            episode_reward += reward
            steps += 1

        if args.agent == 'dqn':
            agent.decay_epsilon()

        if episode % config.LOG_FREQUENCY == 0:
            metrics = f"Episode {episode:5d}/{args.episodes} | Reward: {episode_reward:7.2f}"
            if args.agent == 'dqn':
                metrics += f" | Epsilon: {agent.epsilon:.3f}"
            print(metrics)
            
            
    print("Training Complete!")
    if args.agent == 'dqn':
        agent.epsilon = 0.0

    eval_rewards = []
    best_circuit = None
    best_reward = -float('inf')

    for _ in range(100):
        obs, _ = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            action_mask = env.get_action_mask()

            if args.agent == 'baseline':
                relay_info = env.get_relay_info()
                action = agent.policy(obs, action_mask, relay_info)
            else:
                action = agent.policy(obs, action_mask, env.get_relay_info())

            next_obs, reward, terminated, _, _ = env.step(action)
            obs = next_obs
            episode_reward += reward

        eval_rewards.append(episode_reward)

        if episode_reward > best_reward and env.exit_relay is not None:
            best_reward = episode_reward
            best_circuit = {
                'entry_guard': env.entry_guard,
                'middle_relay': env.middle_relay,
                'exit_relay': env.exit_relay,
                'reward': episode_reward
            }
    
    print("\nEvaluation Results (100 episodes, epsilon=0):")
    print(f" Mean: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f" Best: {np.max(eval_rewards):.2f} | Worst: {np.min(eval_rewards):.2f}")

    if best_circuit is not None:
        entry_guard = env.relays[best_circuit['entry_guard']]
        middle_relay = env.relays[best_circuit['middle_relay']]
        exit_relay = env.relays[best_circuit['exit_relay']]

        print("\nBest Circuit Found:")
        print(f"Entry Guard: #{best_circuit['entry_guard']:3d}: Bandwidth = {entry_guard['bandwidth']:6.2f} MB/s, Latency = {entry_guard['latency']:6.2f} ms")
        print(f"Middle Relay: #{best_circuit['middle_relay']:3d}: Bandwidth = {middle_relay['bandwidth']:6.2f} MB/s, Latency = {middle_relay['latency']:6.2f} ms")
        print(f"Exit Relay: #{best_circuit['exit_relay']:3d}: Bandwidth = {exit_relay['bandwidth']:6.2f} MB/s, Latency = {exit_relay['latency']:6.2f} ms")

        circuit_bandwidth = min(entry_guard['bandwidth'], middle_relay['bandwidth'], exit_relay['bandwidth'])
        circuit_latency = entry_guard['latency'] + middle_relay['latency'] + exit_relay['latency']

        print("\nCircuit Performance")
        print(f"Total Bandwidth: {circuit_bandwidth:.2f} MB/s")
        print(f"Total Latency: {circuit_latency:.2f} ms")
        print(f"Final Reward: {best_circuit['reward']:.2f}")
    else:
        print("\nCircuit Failed")

if __name__ == "__main__":
    main()
