import os
import pathlib
import argparse
import numpy as np
import torch

# Import necessary modules – ensure these packages are in your PYTHONPATH.
from diffusion.utils import *
from corl.algorithms.iql import *
from corl.shared.buffer import *
from corl.shared.logger import *
import composuite

def identify_special_dimensions(data):
    integer_dims = []
    constant_dims = []
    
    for i in range(data.shape[1]):
        column = data[:, i]
        if np.all(np.equal(column, np.round(column))):
            integer_dims.append(i)
        elif np.all(column == column[0]):
            constant_dims.append(i)
    return integer_dims, constant_dims

def process_special_dimensions(synthetic_dataset, integer_dims, constant_dims):
    processed_dataset = {k: v.copy() for k, v in synthetic_dataset.items()}
    
    for key in ['observations', 'next_observations']:
        # Round integer dimensions
        if integer_dims:
            processed_dataset[key][:, integer_dims] = np.round(
                synthetic_dataset[key][:, integer_dims]
            )
        # Round constant dimensions to 2 decimal places
        if constant_dims:
            processed_dataset[key][:, constant_dims] = np.round(
                synthetic_dataset[key][:, constant_dims], 
                decimals=2
            )
    return processed_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="IQL Training Script")
    
    # Data and results paths.
    parser.add_argument("--data_type", type=str, choices=["agent", "synthetic"],
                        default="synthetic", help="Type of dataset to use")
    parser.add_argument("--synthetic_run_id", type=str, default="",
                        help="Synthetic run id (if using synthetic data)")
    # parser.add_argument("--mode", type=str, default="",
    #                     help="Mode for synthetic dataset (e.g., train or test)")
    parser.add_argument("--base_agent_data_path", type=str,
                        default="/Users/shubhankar/Developer/compositional-rl-synth-data/data",
                        help="Path to agent data")
    parser.add_argument("--base_synthetic_data_path", type=str,
                        default="/Users/shubhankar/Developer/compositional-rl-synth-data/cluster_results/Quan/180M/128/test/",
                        help="Path to synthetic data")
    parser.add_argument("--base_results_folder", type=str,
                        default="/Users/shubhankar/Developer/compositional-rl-synth-data/local_results/offline_learning",
                        help="Folder to store training results")
    
    # Task parameters.
    parser.add_argument("--robot", type=str, default="IIWA", help="Robot type")
    parser.add_argument("--obj", type=str, default="Hollowbox", help="Object type")
    parser.add_argument("--obst", type=str, default="None", help="Obstacle type")
    parser.add_argument("--subtask", type=str, default="Shelf", help="Subtask type")
    
    # Training parameters.
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_timesteps", type=int, default=50000, help="Maximum timesteps for training")
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create a unique results folder.
    base_results_path = pathlib.Path(args.base_results_folder)
    idx = 1
    while (base_results_path / f"offline_learning_{args.data_type}_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"offline_learning_{args.data_type}_{idx}"
    print(f"Results folder: {results_folder}")
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize training configuration.
    config = TrainConfig()
    config.seed = args.seed
    config.max_timesteps = args.max_timesteps
    config.n_episodes = args.n_episodes
    config.batch_size = args.batch_size
    config.device = args.device

    

    # Set task parameters.
    robot = args.robot
    obj = args.obj
    obst = args.obst
    subtask = args.subtask

    # Load agent dataset for special-dimension identification.
    env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=True, ignore_done=False)
    agent_dataset = load_single_composuite_dataset(
        base_path=args.base_agent_data_path,
        dataset_type='expert',
        robot=robot, obj=obj,
        obst=obst, task=subtask
    )
    agent_dataset, _ = remove_indicator_vectors(env.modality_dims, transitions_dataset(agent_dataset))
    
    integer_dims, constant_dims = identify_special_dimensions(agent_dataset['observations'])
    print('Integer dimensions:', integer_dims)
    print('Constant dimensions:', constant_dims)
    
    # Load dataset based on the data type.
    if args.data_type == 'synthetic':
        print("Training on synthetic data.")
        synthetic_dataset = load_single_synthetic_dataset(
            base_path=os.path.join(args.base_synthetic_data_path, args.synthetic_run_id), #, args.mode
            robot=robot, obj=obj,
            obst=obst, task=subtask
        )
        synthetic_dataset = process_special_dimensions(synthetic_dataset, integer_dims, constant_dims)
        dataset = synthetic_dataset
    elif args.data_type == 'agent':
        print("Training on agent data.")
        dataset = agent_dataset
    else:
        raise ValueError("Invalid data_type specified. Choose 'agent' or 'synthetic'.")
    
    num_samples = int(dataset['observations'].shape[0])
    print("Samples:", num_samples)
    
    # Environment setup.
    print("Environment name:", args.robot, args.obj, args.obst, args.subtask)
    print("Run name:", args.synthetic_run_id)
    print("Dataset observations shape:", dataset["observations"].shape)
    env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=False, ignore_done=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    
    # Prepare replay buffer.
    replay_buffer = prepare_replay_buffer(
        state_dim=state_dim,
        action_dim=action_dim,
        dataset=dataset,
        num_samples=num_samples,
        device=args.device,
        reward_normalizer=RewardNormalizer(dataset, config.env) if config.normalize_reward else None,
        state_normalizer=StateNormalizer(state_mean, state_std),
    )
    
    max_action = float(env.action_space.high[0])
    logger = Logger(results_folder, seed=args.seed)
    
    # Set seed for reproducibility.
    seed = args.seed
    set_seed(seed, env)

    
    
    # Initialize networks and optimizers.
    print(config.device)
    q_network = TwinQ(state_dim, action_dim, hidden_dim=config.network_width, n_hidden=config.network_depth).to(config.device)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    
    v_network = ValueFunction(state_dim, hidden_dim=config.network_width, n_hidden=config.network_depth).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
    
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action, hidden_dim=config.network_width, n_hidden=config.network_depth)
        if config.iql_deterministic else
        GaussianPolicy(state_dim, action_dim, max_action, hidden_dim=config.network_width, n_hidden=config.network_depth)
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    
    # Bundle training hyperparameters.
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL-specific parameters
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps
    }
    
    print("----------------------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("----------------------------------------------------")
    
    trainer = ImplicitQLearning(**kwargs)
    

    # # Initialize wandb.
    # wandb_project = 'offline_rl_diffusion'
    # #wandb_entity = ''
    # wandb_group = 'corl_training'
    # wandb.init(
    #     project=wandb_project,
    #  #   entity=wandb_entity,
    #     group=wandb_group,
    #     name=results_folder.name,
    # )
    
    # Set the checkpoints path.
    config.checkpoints_path = results_folder
    print("Checkpoints path:", config.checkpoints_path)
    
    # Main training loop.
    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
    
        if t % config.log_every == 0:
            # wandb.log(log_dict, step=trainer.total_it)
            logger.log({'step': trainer.total_it, **log_dict}, mode='train')
    
        # Evaluate episode
        if t % config.eval_freq == 0 or t == config.max_timesteps - 1:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            evaluations.append(eval_score)
            print("------------------------------------------------")
            print(f"Evaluation over {config.n_episodes} episodes: {eval_score:.3f}")
            print("------------------------------------------------")
            if config.checkpoints_path is not None and config.save_checkpoints:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            log_dict = {"Score": eval_score}
            # wandb.log(log_dict, step=trainer.total_it)
            logger.log({'step': trainer.total_it, **log_dict}, mode='eval')
    
    print("State mean:", state_mean.mean(), "State std:", state_std.mean())
    
    def get_weights_norm(model):
        total_norm = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total_norm += param.norm(2).item() ** 2
        return total_norm ** 0.5
    
    print('After:', get_weights_norm(actor))
    
    # Optionally, you can uncomment and use the following visualization code.
    # state = env.reset()
    # env.viewer.set_camera(camera_id=3)
    # for _ in range(1000):
    #     action = actor.act(state)
    #     state, _, done, _ = env.step(action)
    #     if done:
    #         break
    #     env.render()

if __name__ == "__main__":
    print("IQL Training Script")
    main()
    print("Done")
