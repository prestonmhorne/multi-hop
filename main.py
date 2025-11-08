# main.py

import numpy as np
import config
from tor_circuit_env import CircuitEnv
from q_learning import QLearningAgent

def main():
    np.random.seed(config.RANDOM_SEED)

    env = CircuitEnv()

    agent = QLearningAgent(num_relays=config.NUM_RELAYS)

    print("Training...")
    print(f"Episodes: {config.NUM_EPISODES}")
    print(f"Relays: {config.NUM_RELAYS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epsilon: {config.EPSILON} -> {config.MIN_EPSILON}\n")

    for episode in range(config.NUM_EPISODES):
        obs, _ = env.reset()
        terminated = False
        episode_reward = 0
        steps = 0

        while not terminated:
            action = agent.policy(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_reward += reward
            steps += 1

        agent.decay_epsilon()

        if episode % config.LOG_FREQUENCY == 0:
            print(f"Episode {episode:5d}/{config.NUM_EPISODES} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Steps: {steps} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
