# main.py

import numpy as np
import torch
import argparse
import config
from circuit_env import CircuitEnv
from dqn_agent import DQNAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='dqn',
                        choices=['dqn'])
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES)
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CircuitEnv()

    agent = DQNAgent(
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
            action = agent.policy(obs, action_mask)
            next_obs, reward, terminated, _, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_reward += reward
            steps += 1

        agent.decay_epsilon()

        if episode % config.LOG_FREQUENCY == 0:
            metrics = f"Episode {episode:5d}/{args.episodes} | Reward: {episode_reward:7.2f} | Steps: {steps}"
            metrics += f" | Epsilon: {agent.epsilon:.3f}"
            print(metrics)
            
            
    print("Training Complete!")
    print(f"Final Epsilon: {agent.epsilon:.3f}")

    if env.exit_relay is not None:
        entry_guard = env.relays[env.entry_guard]
        middle_relay = env.relays[env.middle_relay]
        exit_relay = env.relays[env.exit_relay]

        print("\nFinal Circuit")   
        print(f"Entry Guard: #{env.entry_guard:3d}: Bandwidth = {entry_guard['bandwidth']:6.2f} MB/s, Latency = {entry_guard['latency']:6.2f} ms")
        print(f"Middle Relay: #{env.middle_relay:3d}: Bandwidth = {middle_relay['bandwidth']:6.2f} MB/s, Latency = {middle_relay['latency']:6.2f} ms")
        print(f"Exit Relay: #{env.exit_relay:3d}: Bandwidth = {exit_relay['bandwidth']:6.2f} MB/s, Latency = {exit_relay['latency']:6.2f} ms")

        circuit_bandwidth = min(entry_guard['bandwidth'], middle_relay['bandwidth'], exit_relay['bandwidth'])
        circuit_latency = entry_guard['latency'] + middle_relay['latency'] + exit_relay['latency']

        print("\nCircuit Performance")
        print(f"Total Bandwidth: {circuit_bandwidth:.2f} MB/s")
        print(f"Total Latency: {circuit_latency:.2f} ms")
        print(f"Final Reward: {episode_reward:.2f}")
    else:
        print("\nCircuit Failed")



if __name__ == "__main__":
    main()
