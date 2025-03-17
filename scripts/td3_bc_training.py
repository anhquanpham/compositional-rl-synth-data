import os
import pathlib
import argparse
import torch

# Import necessary modules – ensure these packages are in your PYTHONPATH.
from diffusion.utils import *
# from corl.algorithms.offline.td3_bc import *
from corl.algorithms.td3_bc import *
from corl.shared.buffer import *
from corl.shared.logger import *
import composuite

def parse_args():
    parser = argparse.ArgumentParser(description="TD3+BC Training Script")

    # Data and results paths.
    parser.add_argument("--data_type", type=str, choices=["agent", "synthetic"],
                        default="synthetic", help="Type of dataset to use")
    parser.add_argument("--synthetic_run_id", type=str, default="comp_diff_23",
                        help="Synthetic run id (if using synthetic data)")
    parser.add_argument("--base_agent_data_path", type=str,
                        default="/home/anhquanpham/projects/data",
                        help="Path to agent data")
    parser.add_argument("--base_synthetic_data_path", type=str,
                        default="/home/anhquanpham/projects/results",
                        help="Path to synthetic data")
    parser.add_argument("--base_results_folder", type=str,
                        default="/home/anhquanpham/projects/results/RL_Run",
                        help="Folder to store training results")

    # Task parameters.
    parser.add_argument("--robot", type=str, default="Jaco", help="Robot type")
    parser.add_argument("--obj", type=str, default="Plate", help="Object type")
    parser.add_argument("--obst", type=str, default="ObjectDoor", help="Obstacle type")
    parser.add_argument("--task", type=str, default="Shelf", help="Task type")

    # Training parameters.
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()

def main():
    
    args = parse_args()
    

    # Create a unique results folder.
    base_results_path = pathlib.Path(args.base_results_folder)
    idx = 1
    while (base_results_path / f"offline_learning_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"offline_learning_{idx}"
    print(f"Results folder: {results_folder}")
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load dataset based on the data type.
    if args.data_type == "agent":
        print("Training on agent data.")
        dataset = load_single_composuite_dataset(
            base_path=args.base_agent_data_path,
            dataset_type="expert",
            robot=args.robot,
            obj=args.obj,
            obst=args.obst,
            task=args.task
        )
        env = composuite.make(args.robot, args.obj, args.obst, args.task,
                                use_task_id_obs=True, ignore_done=False)
        dataset, _ = remove_indicator_vectors(env.modality_dims, transitions_dataset(dataset))
    elif args.data_type == "synthetic":
        print("Training on synthetic data.")
        dataset = load_single_synthetic_dataset(
            base_path=os.path.join(args.base_synthetic_data_path, args.synthetic_run_id),
            robot=args.robot,
            obj=args.obj,
            obst=args.obst,
            task=args.task
        )
    else:
        raise ValueError("Invalid data_type specified. Choose 'agent' or 'synthetic'.")

    
    # Environment setup.
    print("Environment name:", args.robot, args.obj, args.obst, args.task)
    print("Run name:", args.synthetic_run_id)
    env = composuite.make(args.robot, args.obj, args.obst, args.task,
                            use_task_id_obs=False, ignore_done=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    # Training configuration.
    config = TrainConfig()
    print("Dataset observations shape:", dataset["observations"].shape)

    # Prepare replay buffer.
    num_samples = int(0.05 * dataset["observations"].shape[0])
    replay_buffer = prepare_replay_buffer(
        state_dim=state_dim,
        action_dim=action_dim,
        dataset=dataset,
        num_samples=num_samples,
        device=args.device,
        reward_normalizer=RewardNormalizer(dataset, config.env) if config.normalize_reward else None,
        state_normalizer=StateNormalizer(state_mean, state_std)
    )

    max_action = float(env.action_space.high[0])
    logger = Logger(results_folder, seed=args.seed)

    # Set seed for reproducibility.
    set_seed(args.seed, env)

    # Initialize networks and optimizers.
    actor = Actor(state_dim, action_dim, max_action,
                  hidden_dim=config.network_width,
                  n_hidden=config.network_depth).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim,
                      hidden_dim=config.network_width,
                      n_hidden=config.network_depth).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)

    critic_2 = Critic(state_dim, action_dim,
                      hidden_dim=config.network_width,
                      n_hidden=config.network_depth).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    # Bundle training hyperparameters.
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        "alpha": config.alpha,
    }

    print("----------------------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {args.seed}")
    print("----------------------------------------------------")

    # Initialize the trainer.
    trainer = TD3_BC(**kwargs)

    # Main training loop.
    evaluations = []
    #for t in range(int(config.max_timesteps)):
    for t in range(int(50000)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        if t % config.log_every == 0:
            logger.log({'step': trainer.total_it, **log_dict}, mode='train')

        # Periodically evaluate the actor.
        # if t % config.eval_freq == 0 or t == config.max_timesteps - 1:
        if t % config.eval_freq == 0 or t == 50000 - 1:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=args.seed,
            )
            eval_score = eval_scores.mean()
            evaluations.append(eval_score)
            print("------------------------------------------------")
            print(f"Evaluation over {config.n_episodes} episodes: {eval_score:.3f}")
            print("------------------------------------------------")
            if config.checkpoints_path is not None and config.save_checkpoints:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt")
                )
            logger.log({'step': trainer.total_it, "Score": eval_score}, mode='eval')

if __name__ == "__main__":
    
    main()
    