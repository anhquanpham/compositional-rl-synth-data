import os
import pickle

run_name = 'cond_diff_24'
base_folder = '/mnt/kostas-graid/datasets/quanpham/results/diffusion/'
run_folder = os.path.join(base_folder, run_name)

train_task_file = os.path.join(run_folder, 'train_tasks.pkl')
test_task_file = os.path.join(run_folder, 'test_tasks.pkl')

train_output_file = 'train_task_folders_to_move.txt'
test_output_file = 'test_task_folders_to_move.txt'

def get_folders_to_move(task_file, all_folders, run_folder):
    try:
        with open(task_file, 'rb') as f:
            task_tuples = pickle.load(f)
        task_folders = ['_'.join(task) for task in task_tuples]
        folders_to_move = [
            folder for folder in all_folders
            if folder in task_folders and 'samples.npz' in os.listdir(os.path.join(run_folder, folder))
        ]
        return folders_to_move
    except FileNotFoundError:
        print(f"Error: Task file '{task_file}' not found.")
        return []

all_folders = [d for d in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, d))]

# Train
train_folders_to_move = get_folders_to_move(train_task_file, all_folders, run_folder)
train_folders_to_move = train_folders_to_move[:12]
with open(train_output_file, 'w') as f:
    f.write('\n'.join(train_folders_to_move))
print(f"Saved {len(train_folders_to_move)} train folder names to {train_output_file}.")

# Test
test_folders_to_move = get_folders_to_move(test_task_file, all_folders, run_folder)
with open(test_output_file, 'w') as f:
    f.write('\n'.join(test_folders_to_move))
print(f"Saved {len(test_folders_to_move)} test folder names to {test_output_file}.")
