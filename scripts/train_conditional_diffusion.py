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

    parser.add_argument('--compositional', type=str, default='False', help='Use compositional denoiser network.')

    # Environment
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., expert data).')
    parser.add_argument('--experiment_type', type=str, required=True, help='CompoSuite experiment type.', default='default')
    parser.add_argument('--element', type=str, required=False, help='CompoSuite element.')
    parser.add_argument('--num_train', type=int, required=True, help='Number of CompoSuite tasks.')

    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    # W&B config
    parser.add_argument('--wandb_project', type=str, default="offline_rl_diffusion")
    parser.add_argument('--wandb_entity', type=str, default="")
    parser.add_argument('--wandb_group', type=str, default="diffusion_training")
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    base_results_path = pathlib.Path(args.base_results_folder)
    idx = 1
    while (base_results_path / f"non_comp_diff_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"non_comp_diff_{idx}"
    results_folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    if args.experiment_type == 'default':
        # train_tasks, test_tasks = composuite.sample_tasks(experiment_type='default', num_train=args.num_train)
        # test_tasks = test_tasks[:12]
        #train_tasks = [('IIWA', 'Box', 'GoalWall', 'PickPlace')]
        train_tasks = [
            ('IIWA', 'Box', 'GoalWall', 'PickPlace'), #1###############################################UNIQUE
            ('Jaco', 'Plate', 'GoalWall', 'Shelf'),
            ('Panda', 'Plate', 'ObjectWall', 'Shelf'), #####################################UNIQUE
            ('Panda', 'Dumbbell', 'ObjectWall', 'Shelf'),
            ('Jaco', 'Box', 'GoalWall', 'Push'),
            ('Panda', 'Plate', 'ObjectDoor', 'Shelf'),
            ('IIWA', 'Box', 'None', 'Push'),
            ('Jaco', 'Plate', 'None', 'Push'),
            ('Jaco', 'Plate', 'ObjectDoor', 'Shelf'),
            ('Panda', 'Plate', 'ObjectWall', 'PickPlace'), #10
            ('Kinova3', 'Box', 'ObjectDoor', 'PickPlace'),
            ('Kinova3', 'Box', 'GoalWall', 'Push'),
            ('Panda', 'Plate', 'GoalWall', 'Trashcan'),
            ('Panda', 'Dumbbell', 'None', 'Shelf'),
            ('Kinova3', 'Plate', 'ObjectWall', 'PickPlace'),
            ('Panda', 'Box', 'None', 'Push'),
            ('Kinova3', 'Dumbbell', 'ObjectWall', 'Push'),
            ('Jaco', 'Box', 'ObjectWall', 'Trashcan'),
            ('Jaco', 'Hollowbox', 'ObjectWall', 'Shelf'),
            ('Jaco', 'Dumbbell', 'None', 'Trashcan'), #20
            ('Kinova3', 'Dumbbell', 'None', 'Trashcan'),
            ('Panda', 'Dumbbell', 'None', 'Push'),
            ('Jaco', 'Plate', 'ObjectWall', 'Trashcan'),
            ('IIWA', 'Hollowbox', 'ObjectDoor', 'Shelf'),
            ('Jaco', 'Plate', 'ObjectDoor', 'Push'),
            ('Kinova3', 'Plate', 'ObjectDoor', 'Push'),
            ('Kinova3', 'Plate', 'GoalWall', 'PickPlace'),
            ('Panda', 'Hollowbox', 'None', 'Shelf'),
            ('Panda', 'Dumbbell', 'None', 'PickPlace'),
            ('Jaco', 'Hollowbox', 'ObjectWall', 'PickPlace'), #30
            ('IIWA', 'Plate', 'GoalWall', 'PickPlace'),
            ('IIWA', 'Plate', 'ObjectDoor', 'Trashcan'),
            ('Kinova3', 'Hollowbox', 'GoalWall', 'Push'),
            ('Kinova3', 'Dumbbell', 'ObjectDoor', 'Push'),
            ('IIWA', 'Dumbbell', 'ObjectWall', 'PickPlace'),
            ('IIWA', 'Box', 'ObjectDoor', 'Shelf'),
            ('IIWA', 'Box', 'None', 'Trashcan'),
            ('Jaco', 'Box', 'ObjectDoor', 'PickPlace'),
            ('Kinova3', 'Plate', 'None', 'Push'),
            ('IIWA', 'Plate', 'ObjectWall', 'Trashcan'), #40
            ('Kinova3', 'Box', 'ObjectDoor', 'Push'),
            ('Panda', 'Plate', 'ObjectDoor', 'Trashcan'),
            ('Panda', 'Hollowbox', 'None', 'Push'),
            ('Kinova3', 'Hollowbox', 'ObjectWall', 'PickPlace'),
            ('Kinova3', 'Hollowbox', 'ObjectDoor', 'Trashcan'),
            ('Panda', 'Plate', 'GoalWall', 'PickPlace'),
            ('Panda', 'Hollowbox', 'GoalWall', 'Push'),
            ('IIWA', 'Hollowbox', 'None', 'Shelf'),
            ('IIWA', 'Box', 'ObjectWall', 'Push'),
            ('IIWA', 'Plate', 'GoalWall', 'Trashcan'), #50
            ('IIWA', 'Box', 'GoalWall', 'Push'),
            ('Panda', 'Plate', 'None', 'Push'),
            ('Panda', 'Box', 'ObjectWall', 'PickPlace'),
            ('IIWA', 'Dumbbell', 'ObjectDoor', 'PickPlace'),
            ('IIWA', 'Hollowbox', 'GoalWall', 'Shelf'),
            ('Jaco', 'Hollowbox', 'ObjectWall', 'Trashcan'), #56
            ('Kinova3', 'Plate', 'GoalWall', 'Push'),
            ('Kinova3', 'Box', 'ObjectWall', 'PickPlace'),
            ('Panda', 'Box', 'ObjectWall', 'Shelf'),
            ('Kinova3', 'Dumbbell', 'GoalWall', 'Push'),  #60
            ('Kinova3', 'Hollowbox', 'ObjectWall', 'Push'),
            ('Jaco', 'Plate', 'ObjectDoor', 'Trashcan'),
            ('Kinova3', 'Hollowbox', 'GoalWall', 'Shelf'),
            ('Kinova3', 'Hollowbox', 'None', 'PickPlace'),
            ('IIWA', 'Box', 'ObjectWall', 'PickPlace'),
            ('IIWA', 'Box', 'GoalWall', 'Trashcan'),
            ('Panda', 'Hollowbox', 'ObjectDoor', 'Shelf'),
            ('IIWA', 'Plate', 'None', 'Trashcan'),
            ('IIWA', 'Plate', 'ObjectDoor', 'PickPlace'),
            ('Jaco', 'Plate', 'GoalWall', 'PickPlace'), #70
            ('Jaco', 'Dumbbell', 'None', 'PickPlace'),
            ('Panda', 'Dumbbell', 'ObjectDoor', 'Shelf'),
            ('Panda', 'Dumbbell', 'ObjectDoor', 'Trashcan'),
            ('Kinova3', 'Dumbbell', 'ObjectWall', 'PickPlace'),
            ('IIWA', 'Hollowbox', 'ObjectDoor', 'Push'),
            ('Jaco', 'Box', 'None', 'PickPlace'),
            ('Jaco', 'Box', 'ObjectWall', 'Shelf'),
            ('Panda', 'Plate', 'ObjectWall', 'Trashcan'),
            ('Jaco', 'Hollowbox', 'None', 'Push'),
            ('Jaco', 'Dumbbell', 'ObjectDoor', 'Trashcan'), #80
            ('Kinova3', 'Plate', 'None', 'Trashcan'),
            ('IIWA', 'Box', 'None', 'Shelf'),
            ('IIWA', 'Plate', 'ObjectDoor', 'Shelf'),
            ('Kinova3', 'Box', 'None', 'PickPlace'),
            ('IIWA', 'Hollowbox', 'GoalWall', 'Push'),
            ('Panda', 'Plate', 'None', 'PickPlace'),
            ('Jaco', 'Box', 'None', 'Push'),
            ('Panda', 'Dumbbell', 'ObjectWall', 'Trashcan'),
            ('Panda', 'Box', 'ObjectDoor', 'Push'),
            ('Jaco', 'Hollowbox', 'GoalWall', 'Push'), #90
            ('Panda', 'Dumbbell', 'ObjectDoor', 'PickPlace'),
            ('Kinova3', 'Dumbbell', 'ObjectWall', 'Trashcan'),
            ('Panda', 'Hollowbox', 'ObjectDoor', 'PickPlace'),
            ('Kinova3', 'Plate', 'GoalWall', 'Trashcan'),
            ('IIWA', 'Hollowbox', 'None', 'Push'),
            ('Kinova3', 'Plate', 'GoalWall', 'Shelf'),
            ('Kinova3', 'Dumbbell', 'ObjectDoor', 'Shelf'),
            ('Kinova3', 'Plate', 'ObjectWall', 'Shelf'), #98
            ('IIWA', 'Dumbbell', 'GoalWall', 'Push'),
            ('Kinova3', 'Dumbbell', 'GoalWall', 'PickPlace'), #100
            ('Jaco', 'Dumbbell', 'None', 'Shelf'),
            ('Jaco', 'Box', 'None', 'Trashcan'),
            ('Panda', 'Hollowbox', 'ObjectWall', 'Shelf'),
            ('IIWA', 'Hollowbox', 'ObjectDoor', 'PickPlace'),
            ('Kinova3', 'Box', 'ObjectWall', 'Trashcan'),
            ('Panda', 'Box', 'None', 'PickPlace'),
            ('IIWA', 'Hollowbox', 'ObjectWall', 'Trashcan'),
            ('Kinova3', 'Dumbbell', 'None', 'Push'), #108
            ('Jaco', 'Dumbbell', 'GoalWall', 'Push'),
            ('Kinova3', 'Hollowbox', 'ObjectDoor', 'Push'), #110
            ('Jaco', 'Plate', 'ObjectWall', 'Push'),
            ('IIWA', 'Plate', 'None', 'PickPlace'),
            ('Jaco', 'Box', 'ObjectDoor', 'Trashcan'),
            ('Jaco', 'Hollowbox', 'None', 'PickPlace'),
            ('Jaco', 'Hollowbox', 'ObjectDoor', 'Trashcan'),
            ('IIWA', 'Box', 'ObjectWall', 'Trashcan'),
            ('IIWA', 'Hollowbox', 'GoalWall', 'Trashcan'),
            ('Panda', 'Dumbbell', 'GoalWall', 'Shelf'),
            ('Panda', 'Plate', 'GoalWall', 'Push'),
            ('Jaco', 'Hollowbox', 'ObjectDoor', 'Shelf'), #120
            ('IIWA', 'Dumbbell', 'ObjectWall', 'Shelf'),
            ('IIWA', 'Box', 'ObjectWall', 'Shelf'),
            ('IIWA', 'Box', 'ObjectDoor', 'PickPlace'),
            ('IIWA', 'Hollowbox', 'None', 'Trashcan'), 
            ('Jaco', 'Box', 'GoalWall', 'Trashcan'),
            ('Kinova3', 'Hollowbox', 'GoalWall', 'PickPlace'),
            ('Kinova3', 'Dumbbell', 'GoalWall', 'Trashcan'),
            ('IIWA', 'Plate', 'None', 'Push'),
            ('Jaco', 'Dumbbell', 'GoalWall', 'Trashcan'),
            ('Kinova3', 'Hollowbox', 'None', 'Trashcan'), #130 ################################################UNIQUE
            ('Panda', 'Box', 'ObjectWall', 'Trashcan'),
            ('Jaco', 'Plate', 'GoalWall', 'Push'),
            ('Panda', 'Box', 'ObjectDoor', 'Trashcan'),
            ('Kinova3', 'Plate', 'ObjectDoor', 'Trashcan'),
            ('Kinova3', 'Plate', 'None', 'Shelf'),
            ('Jaco', 'Dumbbell', 'ObjectWall', 'Shelf'),
            ('IIWA', 'Plate', 'GoalWall', 'Push'),
            ('Panda', 'Hollowbox', 'GoalWall', 'Shelf'),
            ('Panda', 'Dumbbell', 'GoalWall', 'Trashcan'),
            ('Panda', 'Box', 'None', 'Shelf'), #140
            ('IIWA', 'Plate', 'ObjectWall', 'Shelf'),
            ('Jaco', 'Dumbbell', 'ObjectDoor', 'Push'),                                             ##UNIQUE
            ('IIWA', 'Dumbbell', 'ObjectWall', 'Trashcan'),
            ('Jaco', 'Plate', 'ObjectDoor', 'PickPlace'),
            ('Panda', 'Plate', 'None', 'Trashcan'),
            ('Jaco', 'Dumbbell', 'ObjectDoor', 'Shelf'),
            ('Panda', 'Plate', 'ObjectDoor', 'Push'),
            ('IIWA', 'Dumbbell', 'ObjectDoor', 'Push'),
            ('Panda', 'Dumbbell', 'GoalWall', 'PickPlace'),
            ('Panda', 'Dumbbell', 'ObjectDoor', 'Push'), #150
            ('Kinova3', 'Dumbbell', 'None', 'Shelf'),
            ('IIWA', 'Hollowbox', 'ObjectWall', 'Push'),
            ('IIWA', 'Box', 'GoalWall', 'Shelf'),
            ('IIWA', 'Dumbbell', 'GoalWall', 'Shelf'),
            ('IIWA', 'Dumbbell', 'GoalWall', 'Trashcan'),
            ('Jaco', 'Box', 'GoalWall', 'Shelf'),
            ('IIWA', 'Plate', 'None', 'Shelf'),
            ('Jaco', 'Hollowbox', 'ObjectDoor', 'PickPlace'),
            ('Kinova3', 'Plate', 'None', 'PickPlace'),
            ('Panda', 'Hollowbox', 'ObjectWall', 'Push'), #160
            ('Kinova3', 'Box', 'ObjectDoor', 'Shelf'),
            ('Panda', 'Hollowbox', 'ObjectDoor', 'Trashcan'),
            ('Kinova3', 'Plate', 'ObjectWall', 'Trashcan'),
            ('Kinova3', 'Dumbbell', 'None', 'PickPlace'),
            ('IIWA', 'Hollowbox', 'ObjectDoor', 'Trashcan'),
            ('Jaco', 'Box', 'None', 'Shelf'),
            ('Kinova3', 'Box', 'None', 'Trashcan'),
            ('IIWA', 'Box', 'ObjectDoor', 'Trashcan'),
            ('Panda', 'Plate', 'ObjectWall', 'Push'),
            ('IIWA', 'Dumbbell', 'ObjectWall', 'Push'),  #170
            ('Kinova3', 'Hollowbox', 'ObjectDoor', 'Shelf'),
            ('IIWA', 'Dumbbell', 'None', 'Trashcan'),
            ('Kinova3', 'Hollowbox', 'None', 'Shelf'),
            ('Panda', 'Hollowbox', 'ObjectWall', 'PickPlace'),
            ('Kinova3', 'Plate', 'ObjectDoor', 'PickPlace'),
            ('Panda', 'Dumbbell', 'ObjectWall', 'PickPlace'),
            ('Panda', 'Hollowbox', 'GoalWall', 'PickPlace'),
            ('Kinova3', 'Box', 'GoalWall', 'Trashcan'),
            ('Panda', 'Hollowbox', 'GoalWall', 'Trashcan'),
            ('Panda', 'Hollowbox', 'ObjectWall', 'Trashcan'), #180
            ('Kinova3', 'Dumbbell', 'ObjectWall', 'Shelf'),
            ('Kinova3', 'Hollowbox', 'GoalWall', 'Trashcan'), #182
            # ('Jaco', 'Hollowbox', 'ObjectWall', 'Push'),
            # ('Kinova3', 'Hollowbox', 'ObjectDoor', 'PickPlace'),
            # ('Jaco', 'Box', 'ObjectWall', 'PickPlace'), 
            # ('Panda', 'Box', 'None', 'Trashcan'),
            # ('Jaco', 'Hollowbox', 'GoalWall', 'Trashcan'),
            # ('Panda', 'Plate', 'GoalWall', 'Shelf'),
            # ('Kinova3', 'Box', 'None', 'Push'),
            # ('IIWA', 'Dumbbell', 'ObjectDoor', 'Trashcan'), #190
            # ('Kinova3', 'Box', 'GoalWall', 'PickPlace'),
            # ('Panda', 'Plate', 'None', 'Shelf'),
            # ('Panda', 'Dumbbell', 'None', 'Trashcan'),
            # ('IIWA', 'Dumbbell', 'ObjectDoor', 'Shelf'),
            # ('Panda', 'Dumbbell', 'GoalWall', 'Push'),
            # ('Jaco', 'Hollowbox', 'None', 'Shelf'),
            # ('Jaco', 'Dumbbell', 'GoalWall', 'Shelf'),
            # ('Jaco', 'Dumbbell', 'None', 'Push'),
            # ('Jaco', 'Dumbbell', 'ObjectWall', 'Push'),
            # ('Jaco', 'Dumbbell', 'ObjectWall', 'Trashcan'), #200
            # ('Kinova3', 'Hollowbox', 'ObjectWall', 'Trashcan'),
            # ('Kinova3', 'Dumbbell', 'ObjectDoor', 'PickPlace'),
            # ('Panda', 'Box', 'ObjectWall', 'Push'),
            # ('Jaco', 'Plate', 'ObjectWall', 'Shelf'),
            # ('Jaco', 'Dumbbell', 'ObjectDoor', 'PickPlace'),
            # ('Panda', 'Plate', 'ObjectDoor', 'PickPlace'),
            # ('IIWA', 'Plate', 'GoalWall', 'Shelf'),
            # ('Jaco', 'Dumbbell', 'GoalWall', 'PickPlace'),
            # ('IIWA', 'Plate', 'ObjectDoor', 'Push'),
            # ('Kinova3', 'Dumbbell', 'GoalWall', 'Shelf'), #210
            # ('Jaco', 'Hollowbox', 'GoalWall', 'Shelf'),
            # ('IIWA', 'Dumbbell', 'GoalWall', 'PickPlace'),
            # ('IIWA', 'Dumbbell', 'None', 'Shelf'),
            # ('Panda', 'Hollowbox', 'None', 'PickPlace'),
            # ('Kinova3', 'Plate', 'ObjectDoor', 'Shelf'),
            # ('IIWA', 'Dumbbell', 'None', 'Push'),
            # ('Jaco', 'Plate', 'GoalWall', 'Trashcan'),
            # ('Jaco', 'Plate', 'None', 'Shelf'),
            # ('Panda', 'Box', 'ObjectDoor', 'Shelf'),
            # ('Kinova3', 'Box', 'ObjectWall', 'Shelf'), #220
            # ('Jaco', 'Box', 'GoalWall', 'PickPlace'),
            # ('Jaco', 'Plate', 'None', 'PickPlace'),
            # ('IIWA', 'Hollowbox', 'ObjectWall', 'PickPlace'),
            # ('Jaco', 'Box', 'ObjectDoor', 'Shelf') #224
        ]

        test_tasks = [
            ('Kinova3', 'Box', 'GoalWall', 'Shelf'), #1
            ('Jaco', 'Box', 'ObjectDoor', 'Push'),
            # ('IIWA', 'Box', 'None', 'PickPlace'),
            # ('Kinova3', 'Box', 'ObjectDoor', 'Trashcan'),
            # ('Panda', 'Hollowbox', 'ObjectDoor', 'Push'),
            # ('Jaco', 'Hollowbox', 'ObjectDoor', 'Push'),
            # ('Kinova3', 'Box', 'ObjectWall', 'Push'),
            # ('Kinova3', 'Hollowbox', 'None', 'Push'),
            # ('IIWA', 'Plate', 'ObjectWall', 'PickPlace'),
            # ('Panda', 'Dumbbell', 'ObjectWall', 'Push'), #10
            # ('Panda', 'Box', 'GoalWall', 'PickPlace'),
            ('Kinova3', 'Dumbbell', 'ObjectDoor', 'Trashcan'),
            # ('IIWA', 'Hollowbox', 'None', 'PickPlace'),
            # ('Panda', 'Box', 'GoalWall', 'Push'),
            # ('IIWA', 'Plate', 'ObjectWall', 'Push'),
            # ('IIWA', 'Hollowbox', 'GoalWall', 'PickPlace'),
            # ('Panda', 'Box', 'GoalWall', 'Shelf'),
            # ('Jaco', 'Hollowbox', 'None', 'Trashcan'),
            # ('Jaco', 'Hollowbox', 'GoalWall', 'PickPlace'),
            # ('Jaco', 'Plate', 'None', 'Trashcan'), #20
            ('IIWA', 'Dumbbell', 'None', 'PickPlace'),
            ('IIWA', 'Hollowbox', 'ObjectWall', 'Shelf'),
            ('Panda', 'Box', 'GoalWall', 'Trashcan'),
            ('Jaco', 'Box', 'ObjectWall', 'Push'),
            # ('Kinova3', 'Plate', 'ObjectWall', 'Push'),
            ('Kinova3', 'Hollowbox', 'ObjectWall', 'Shelf'),
            ('Jaco', 'Plate', 'ObjectWall', 'PickPlace'),
            # ('Jaco', 'Dumbbell', 'ObjectWall', 'PickPlace'),
            # ('Kinova3', 'Box', 'None', 'Shelf'),
            ('Panda', 'Hollowbox', 'None', 'Trashcan'), #30
            ('IIWA', 'Box', 'ObjectDoor', 'Push'),
            ('Panda', 'Box', 'ObjectDoor', 'PickPlace')] #32

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

    compositional = args.compositional == 'True'
    diffusion = construct_diffusion_model(inputs=inputs, compositional=compositional, cond_dim=indicators.shape[1])

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=results_folder.name,
    )

    # Trainer
    trainer = Trainer(diffusion, dataset, results_folder=str(results_folder))
    trainer.train()

    test_tasks = test_tasks[:12]
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


    train_tasks = train_tasks[:12]
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

    # test_tasks = test_tasks[:12]
    # for robot, obj, obst, subtask in test_tasks:
    #     print('Generating synthetic data for test tasks:', robot, obj, obst, subtask)
    #     subtask_folder = results_folder / f"{robot}_{obj}_{obst}_{subtask}"
    #     retry_count = 0
    #     while not subtask_folder.exists():
    #         try:
    #             subtask_folder.mkdir(parents=True, exist_ok=True)
    #         except Exception as exception:
    #             retry_count += 1
    #             if retry_count >= 5:
    #                 raise RuntimeError(f"Failed to create directory {subtask_folder}.") from exception
    #             time.sleep(1)  # wait before retrying

    #     subtask_indicator = get_task_indicator(robot, obj, obst, subtask)
    #     generator = SimpleDiffusionGenerator(env=representative_env, ema_model=trainer.ema.ema_model)
    #     obs, actions, rewards, next_obs, terminals = generator.sample(num_samples=1000000, cond=subtask_indicator)

    #     if not subtask_folder.exists():
    #         print(f'Folder missing unexpectedly: {subtask_folder}')
    #         subtask_folder.mkdir(parents=True, exist_ok=True)

    #     np.savez_compressed(
    #         subtask_folder / 'samples.npz',
    #         observations=obs,
    #         actions=actions,
    #         rewards=rewards,
    #         next_observations=next_obs,
    #         terminals=terminals
    #     )