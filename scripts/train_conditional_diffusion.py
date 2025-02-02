import argparse
import pathlib
import time
import torch
import wandb
import numpy as np
import pickle
import gin
from diffusion.utils import *
from diffusion.elucidated_diffusion import Trainer
from diffusion.train_diffuser import SimpleDiffusionGenerator
import composuite

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_data_path', type=str, required=True, help='Base path to datasets.')
    parser.add_argument('--base_results_folder', type=str, required=True, help='Base path to results.')

    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/diffusion.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[], help='Additional gin parameters.')

    # Environment
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., expert data).')
    parser.add_argument('--experiment_type', type=str, required=True, help='CompoSuite experiment type.', default='default')
    parser.add_argument('--element', type=str, required=False, help='CompoSuite element.')
    parser.add_argument('--num_train', type=int, required=True, help='Number of CompoSuite tasks.')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    # W&B config
    parser.add_argument('--wandb_project', type=str, default="offline_rl_diffusion")
    #parser.add_argument('--wandb_entity', type=str, default="")
    parser.add_argument('--wandb_group', type=str, default="diffusion_training")
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    base_results_path = pathlib.Path(args.base_results_folder)
    idx = 1
    while (base_results_path / f"cond_diff_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"cond_diff_{idx}"
    results_folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    if args.experiment_type == 'default':
        train_tasks, test_tasks = composuite.sample_tasks(experiment_type='default', num_train=args.num_train)
        test_tasks = test_tasks[:12]
    if args.experiment_type == 'smallscale':
        element = args.element
        train_tasks, test_tasks = composuite.sample_tasks(experiment_type='smallscale', 
                                                 smallscale_elem=args.element, num_train=args.num_train)
    if args.experiment_type == 'holdout':
        train_tasks, test_tasks = composuite.sample_tasks(experiment_type='holdout', 
                                                 holdout_elem=args.element, num_train=args.num_train)
        
    with open(results_folder / "train_tasks.pkl", 'wb') as file:
        pickle.dump(train_tasks, file)

    with open(results_folder / "test_tasks.pkl", 'wb') as file:
        pickle.dump(test_tasks, file)

    representative_task = train_tasks[0]
    robot, obj, obst, subtask = representative_task
    representative_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=False, ignore_done=False)
    representative_indicators_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=True, ignore_done=False)
    modality_dims = representative_indicators_env.modality_dims

    task_indicators_dict = {}

    num_samples = 0
    all_inputs = []
    all_indicators = []
    for task in tqdm(train_tasks, desc="Loading data"):
        robot, obj, obst, subtask = task
        dataset = load_single_composuite_dataset(args.base_data_path, args.dataset_type, robot, obj, obst, subtask)
        dataset = transitions_dataset(dataset)
        dataset, indicators = remove_indicator_vectors(modality_dims, dataset)
        task_indicators_dict[task] = indicators[0, :]
        num_samples += dataset['observations'].shape[0]
        inputs = make_inputs(dataset)
        all_inputs.append(inputs)
        all_indicators.append(indicators)

    num_features = all_inputs[0].shape[1]
    num_indicators = all_indicators[0].shape[1]
    all_inputs_matrix = np.empty((num_samples, num_features), dtype=all_inputs[0].dtype)
    all_indicators_matrix = np.empty((num_samples, num_indicators), dtype=all_indicators[0].dtype)

    current_index = 0
    for inputs, indicators in tqdm(zip(all_inputs, all_indicators), desc="Filling processed inputs and indicators matrices"):
        all_inputs_matrix[current_index:current_index + inputs.shape[0]] = inputs
        all_indicators_matrix[current_index:current_index + indicators.shape[0]] = indicators
        current_index += inputs.shape[0]

    inputs = torch.from_numpy(all_inputs_matrix).float()
    indicators = torch.from_numpy(all_indicators_matrix).float()
    dataset = torch.utils.data.TensorDataset(inputs, indicators)

    diffusion = construct_diffusion_model(inputs=inputs, cond_dim=indicators.shape[1])

    wandb.init(
        project=args.wandb_project,
        #entity=args.wandb_entity,
        group=args.wandb_group,
        name=results_folder.name,
    )

    # Trainer
    trainer = Trainer(diffusion, dataset, results_folder=str(results_folder))
    trainer.train()

    for robot, obj, obst, subtask in test_tasks:
        print('Generating synthetic data for test tasks:', robot, obj, obst, subtask)
        subtask_folder = results_folder / f"{robot}_{obj}_{obst}_{subtask}"
        retry_count = 0
        while not subtask_folder.exists():
            try:
                subtask_folder.mkdir(parents=True, exist_ok=True)
            except Exception as exception:
                retry_count += 1
                if retry_count >= 5:
                    raise RuntimeError(f"Failed to create directory {subtask_folder}.") from exception
                time.sleep(1)  # wait before retrying

        subtask_indicator = get_task_indicator(robot, obj, obst, subtask)
        generator = SimpleDiffusionGenerator(env=representative_env, ema_model=trainer.ema.ema_model)
        obs, actions, rewards, next_obs, terminals = generator.sample(num_samples=1000000, cond=subtask_indicator)

        if not subtask_folder.exists():
            print(f'Folder missing unexpectedly: {subtask_folder}')
            subtask_folder.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            subtask_folder / 'samples.npz',
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            terminals=terminals
        )

    for idx, (robot, obj, obst, subtask) in enumerate(train_tasks):
        print('Generating synthetic data for train tasks:', robot, obj, obst, subtask)
        subtask_folder = results_folder / f"{robot}_{obj}_{obst}_{subtask}"
        retry_count = 0
        while not subtask_folder.exists():
            try:
                subtask_folder.mkdir(parents=True, exist_ok=True)
            except Exception as exception:
                retry_count += 1
                if retry_count >= 5:
                    raise RuntimeError(f"Failed to create directory {subtask_folder}.") from exception
                time.sleep(1)  # wait before retrying

        subtask_indicator = task_indicators_dict[(robot, obj, obst, subtask)]
        generator = SimpleDiffusionGenerator(env=representative_env, ema_model=trainer.ema.ema_model)
        obs, actions, rewards, next_obs, terminals = generator.sample(num_samples=1000000, cond=subtask_indicator)

        if not subtask_folder.exists():
            print(f'Folder missing unexpectedly: {subtask_folder}')
            subtask_folder.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            subtask_folder / 'samples.npz',
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            terminals=terminals
        )
