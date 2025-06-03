import os
import torch
import argparse
import composuite
import numpy as np
from corl.algorithms.td3_bc import Actor
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Visualization for Trained TD3+BC Policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint (.pt) file containing the TD3+BC actor state_dict")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on, e.g., 'cpu' or 'cuda'")
    parser.add_argument("--robot", type=str, default="Jaco", help="Robot type")
    parser.add_argument("--obj", type=str, default="Plate", help="Object type")
    parser.add_argument("--obst", type=str, default="ObjectDoor", help="Obstacle type")
    parser.add_argument("--task", type=str, default="Shelf", help="Task type")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()
    

def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create environment with rendering enabled.
    env = composuite.make(args.robot, args.obj, args.obst, args.task,
                            use_task_id_obs=False, has_renderer=True, ignore_done=False)
    
    # Set seeds for reproducibility.
    set_seed(args.seed, env)
    
    # Extract observation and action dimensions.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Instantiate the actor with the correct architecture.
    actor = Actor(state_dim, action_dim, max_action, hidden_dim=256, n_hidden=2).to(device)
    
    # Load the saved checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    
    # Reset the environment and set a camera view if supported.
    state = env.reset()
    try:
        env.viewer.set_camera(camera_id=0)
    except Exception as e:
        print("Could not set camera view:", e)
        
    # Initialize an accumulator for total rewards.
    total_reward = 0.0
    
    low, high = env.action_spec

    
    # Run a rollout for a fixed number of steps or until the episode ends.
    for _ in range(5000):
        action = actor.act(state, device=args.device)
        state, reward, done, info = env.step(action)
        total_reward += reward  # Accumulate rewards.
        env.render()
        if done:
            break
    print("Total reward:", total_reward)
    env.close()

if __name__ == '__main__':
    # python td3bc_inference.py --checkpoint "/path/to/your/checkpoint.pt" --device "cuda" --robot "Kinova3" --obj "Hollowbox" --obst "ObjectWall" --task "Shelf"
    main()

