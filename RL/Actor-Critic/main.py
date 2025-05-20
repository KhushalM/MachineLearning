from train import Trainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Actor-Critic Training")
    parser.add_argument("--env_id", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--render", type=bool, default=False, help="Render environment")
    args = parser.parse_args()
    trainer = Trainer(args.env_id, args.episodes, args.render)
    trainer.train()
    trainer.plot_rewards_losses()
    