"""
Denoiser networks for diffusion.
Code adapts heavily from https://github.com/conglu1997/SynthER.

This version includes optimized versions of:
  - _VectorEncoder (using nn.Sequential when possible)
  - Encoder (with proper abstract properties)
  - CompositionalMLP (with vectorized batch operations in forward)
  - CompositionalResidualBlock and CompositionalResidualMLP (with additional vectorization)
"""

"""
THIS VERSION IS USING SEPARATE MLPS FOR COMPONENTS
"""

import math
from typing import Optional, Sequence, Dict, Any
from abc import ABCMeta, abstractmethod

import gin
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
from einops import rearrange


##########################
# Utility functions
##########################
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def fanin_init(tensor):
    """Initialize the weights of a layer with fan-in initialization."""
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


##########################
# Positional Embeddings (unchanged)
##########################
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Following @crowsonkb's lead with random (or learned) sinusoidal positional embedding.
    """
    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


##########################
# Residual MLP (Original)
##########################
class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        self.ln = nn.LayerNorm(dim_in) if layer_norm else nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))

class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int, activation: str = "relu", layer_norm: bool = False):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
            nn.LayerNorm(width) if layer_norm else nn.Identity(),
        )
        self.activation = getattr(F, activation)
        self.final_linear = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.activation(self.network(x)))

@gin.configurable
class ResidualMLPDenoiser(nn.Module):
    def __init__(
        self,
        d_in: int,
        dim_t: int = 128,
        mlp_width: int = 1024,
        num_layers: int = 6,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = True,
        learned_sinusoidal_dim: int = 16,
        activation: str = "relu",
        layer_norm: bool = True,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.residual_mlp = ResidualMLP(
            input_dim=dim_t,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )
        if cond_dim is not None:
            self.proj = nn.Linear(d_in + cond_dim, dim_t)
            self.conditional = True
        else:
            self.proj = nn.Linear(d_in, dim_t)
            self.conditional = False

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond=None) -> torch.Tensor:
        if self.conditional:
            assert cond is not None
            x = torch.cat((x, cond), dim=-1)
        time_embed = self.time_mlp(timesteps)
        x = self.proj(x) + time_embed
        return self.residual_mlp(x)


##########################
# Optimized CompositionalResidualBlock and CompositionalResidualMLP
##########################

# Given StateComponentBlock for each state component
class StateComponentBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        self.ln = nn.LayerNorm(dim_in) if layer_norm else nn.Identity()
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(self.ln(x)))

# SeparateStateMLPs processes each state component separately
class SeparateStateMLPs(nn.Module):
    def __init__(self, observation_shape):
        """
        Args:
            encoder_kwargs: (unused here but kept for compatibility)
            observation_shape: A tuple representing the input state vector shape.
                               We expect the state vector to have length 77 * pf,
                               where pf is the projection factor. The state vector
                               is assumed to be the concatenation of:
                                 - object-state: 14 dims (scaled by pf)
                                 - obstacle-state: 14 dims (scaled by pf)
                                 - goal-state: 17 dims (scaled by pf)
                                 - robot0_proprio-state: 32 dims (scaled by pf)
        """
        super().__init__()
        state_dim = observation_shape[0]
        if state_dim % 77 != 0:
            raise ValueError("The state dimension must be a multiple of 77.")
        self.pf = state_dim // 77

        # Create a ResidualBlock for each state component
        self.object_block   = StateComponentBlock(dim_in=14 * self.pf, dim_out=14 * self.pf)
        self.obstacle_block = StateComponentBlock(dim_in=14 * self.pf, dim_out=14 * self.pf)
        self.goal_block     = StateComponentBlock(dim_in=17 * self.pf, dim_out=17 * self.pf)
        self.proprio_block  = StateComponentBlock(dim_in=32 * self.pf, dim_out=32 * self.pf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input state tensor of shape [batch, 77 * pf].
        Returns:
            The processed state tensor after applying separate residual blocks
            to each state component and concatenating the outputs.
        """
        pf = self.pf
        # Slice the state vector into its four components
        object_state    = x[:, :14 * pf]
        obstacle_state  = x[:, 14 * pf:28 * pf]
        goal_state      = x[:, 28 * pf:45 * pf]
        proprio_state   = x[:, 45 * pf:77 * pf]

        # Process each component with its corresponding residual block
        object_out    = self.object_block(object_state)
        obstacle_out  = self.obstacle_block(obstacle_state)
        goal_out      = self.goal_block(goal_state)
        proprio_out   = self.proprio_block(proprio_state)

        # Concatenate all processed outputs into one state vector
        return torch.cat([object_out, obstacle_out, goal_out, proprio_out], dim=-1)
        


##########################
# Optimized CompositionalResidualBlock and CompositionalResidualMLP
##########################
class CompositionalResidualBlock(nn.Module):
    def __init__(self, projection_factor, observation_shape, layer_norm):
        super().__init__()
        self.projection_factor = projection_factor
        self.state_compositional_encoder = SeparateStateMLPs(
            # encoder_kwargs=encoder_kwargs,
            observation_shape=observation_shape,
        )
        self.next_state_compositional_encoder = SeparateStateMLPs(
            # encoder_kwargs=encoder_kwargs,
            observation_shape=observation_shape,
        )
        # Now the total input dimension is 164 * pf (no one-hot)
        self.ln = nn.LayerNorm(164 * self.projection_factor) if layer_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.mlp_linear = nn.Linear(164 * self.projection_factor, 164 * self.projection_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pf = self.projection_factor
        # Extract components from x (now x has shape [batch, 164 * pf])
        current_state = x[:, :77 * pf]         # [batch, 77*pf]
        action = x[:, 77 * pf:85 * pf]           # [batch, 8*pf]
        reward = x[:, 85 * pf:86 * pf]           # [batch, 1*pf]
        next_state = x[:, 86 * pf:163 * pf]        # [batch, 77*pf]
        terminal_flag = x[:, 163 * pf:164 * pf]    # [batch, 1*pf]

        # Process current and next states using separate encoders
        current_state_emb = self.state_compositional_encoder(current_state)
        next_state_emb = self.next_state_compositional_encoder(next_state)

        # Form the learned residual and add it back to the input
        learned_residual = torch.cat([current_state_emb, action, reward, next_state_emb, terminal_flag], dim=-1)
        learned_residual = self.mlp_linear(self.activation(self.ln(learned_residual)))
        first_part = x[:, :164 * pf] + learned_residual
        return first_part

class CompositionalResidualMLP(nn.Module):
    def __init__(self, projection_factor, observation_shape, depth: int, activation: str = "relu", layer_norm: bool = False):
        super().__init__()
        self.projection_factor = projection_factor
        self.network = nn.Sequential(
            *[CompositionalResidualBlock(self.projection_factor, observation_shape, layer_norm)
              for _ in range(depth)]
        )
        self.activation = getattr(F, activation)


        # Independent projection layers for each component
        self.state_proj_object = nn.Linear(14, 14 * self.projection_factor)
        self.state_proj_obstacle = nn.Linear(14, 14 * self.projection_factor)
        self.state_proj_goal = nn.Linear(17, 17 * self.projection_factor)
        self.state_proj_proprio = nn.Linear(32, 32 * self.projection_factor)

        # Independent projection layers for each component
        self.next_state_proj_object = nn.Linear(14, 14 * self.projection_factor)
        self.next_state_proj_obstacle = nn.Linear(14, 14 * self.projection_factor)
        self.next_state_proj_goal = nn.Linear(17, 17 * self.projection_factor)
        self.next_state_proj_proprio = nn.Linear(32, 32 * self.projection_factor)

        
        self.proj_action = nn.Linear(8, 8 * self.projection_factor)
        self.proj_reward = nn.Linear(1, 1 * self.projection_factor)
        self.proj_terminal_flag = nn.Linear(1, 1 * self.projection_factor)

        # Layer norms per component
        self.object_layer_norm = nn.LayerNorm(14 * self.projection_factor) if layer_norm else nn.Identity()
        self.obstacle_layer_norm = nn.LayerNorm(14 * self.projection_factor) if layer_norm else nn.Identity()
        self.goal_layer_norm = nn.LayerNorm(17 * self.projection_factor) if layer_norm else nn.Identity()
        self.proprio_layer_norm = nn.LayerNorm(32 * self.projection_factor) if layer_norm else nn.Identity()
        self.action_layer_norm = nn.LayerNorm(8 * self.projection_factor) if layer_norm else nn.Identity()
        self.reward_layer_norm = nn.LayerNorm(1 * self.projection_factor) if layer_norm else nn.Identity()
        self.terminal_flag_layer_norm = nn.LayerNorm(1 * self.projection_factor) if layer_norm else nn.Identity()

        # Final projection layers to map back to original dimensions
        self.state_final_proj_object = nn.Linear(14 * self.projection_factor, 14)
        self.state_final_proj_obstacle = nn.Linear(14 * self.projection_factor, 14)
        self.state_final_proj_goal = nn.Linear(17 * self.projection_factor, 17)
        self.state_final_proj_proprio = nn.Linear(32 * self.projection_factor, 32)

        # Final projection layers to map back to original dimensions
        self.next_state_final_proj_object = nn.Linear(14 * self.projection_factor, 14)
        self.next_state_final_proj_obstacle = nn.Linear(14 * self.projection_factor, 14)
        self.next_state_final_proj_goal = nn.Linear(17 * self.projection_factor, 17)
        self.next_state_final_proj_proprio = nn.Linear(32 * self.projection_factor, 32)


        self.final_proj_action = nn.Linear(8 * self.projection_factor, 8)
        self.final_proj_reward = nn.Linear(1 * self.projection_factor, 1)
        self.final_proj_terminal_flag = nn.Linear(1 * self.projection_factor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pf = self.projection_factor
        # Now input x has shape [batch, 164] (no one-hot)
        current_state = x[:, :77]         # [batch, 77]
        action = x[:, 77:85]              # [batch, 8]
        reward = x[:, 85:86]              # [batch, 1]
        next_state = x[:, 86:163]         # [batch, 77]
        terminal_flag = x[:, 163:164]     # [batch, 1]

        # Further split current and next state into semantic parts
        cs_obj = current_state[:, :14]
        cs_obst = current_state[:, 14:28]
        cs_goal = current_state[:, 28:45]
        cs_prop = current_state[:, 45:77]

        ns_obj = next_state[:, :14]
        ns_obst = next_state[:, 14:28]
        ns_goal = next_state[:, 28:45]
        ns_prop = next_state[:, 45:77]

        proj_cs_obj = self.state_proj_object(cs_obj)
        proj_cs_obst = self.state_proj_obstacle(cs_obst)
        proj_cs_goal = self.state_proj_goal(cs_goal)
        proj_cs_prop = self.state_proj_proprio(cs_prop)

        proj_ns_obj = self.next_state_proj_object(ns_obj)
        proj_ns_obst = self.next_state_proj_obstacle(ns_obst)
        proj_ns_goal = self.next_state_proj_goal(ns_goal)
        proj_ns_prop = self.next_state_proj_proprio(ns_prop)

        proj_current_state = torch.cat([proj_cs_obj, proj_cs_obst, proj_cs_goal, proj_cs_prop], dim=-1)
        proj_next_state = torch.cat([proj_ns_obj, proj_ns_obst, proj_ns_goal, proj_ns_prop], dim=-1)

        proj_action = self.proj_action(action)
        proj_reward = self.proj_reward(reward)
        proj_terminal_flag = self.proj_terminal_flag(terminal_flag)

        # Form the projected input without one-hot
        projected_input = torch.cat([proj_current_state, proj_action, proj_reward, proj_next_state, proj_terminal_flag], dim=-1)
        repeated_residual_output = self.network(projected_input)

        # Precomputed boundaries for each component (note: these are multiplied by pf)
        boundaries = [b * pf for b in [0, 14, 28, 45, 77, 85, 86, 100, 114, 131, 163, 164]]
        out_state_obj = self.state_final_proj_object(self.activation(self.object_layer_norm(repeated_residual_output[:, boundaries[0]:boundaries[1]])))
        out_state_obst = self.state_final_proj_obstacle(self.activation(self.obstacle_layer_norm(repeated_residual_output[:, boundaries[1]:boundaries[2]])))
        out_state_goal = self.state_final_proj_goal(self.activation(self.goal_layer_norm(repeated_residual_output[:, boundaries[2]:boundaries[3]])))
        out_state_prop = self.state_final_proj_proprio(self.activation(self.proprio_layer_norm(repeated_residual_output[:, boundaries[3]:boundaries[4]])))
        out_action = self.final_proj_action(self.activation(self.action_layer_norm(repeated_residual_output[:, boundaries[4]:boundaries[5]])))
        out_reward = self.final_proj_reward(self.activation(self.reward_layer_norm(repeated_residual_output[:, boundaries[5]:boundaries[6]])))
        out_ns_obj = self.next_state_final_proj_object(self.activation(self.object_layer_norm(repeated_residual_output[:, boundaries[6]:boundaries[7]])))
        out_ns_obst = self.next_state_final_proj_obstacle(self.activation(self.obstacle_layer_norm(repeated_residual_output[:, boundaries[7]:boundaries[8]])))
        out_ns_goal = self.next_state_final_proj_goal(self.activation(self.goal_layer_norm(repeated_residual_output[:, boundaries[8]:boundaries[9]])))
        out_ns_prop = self.next_state_final_proj_proprio(self.activation(self.proprio_layer_norm(repeated_residual_output[:, boundaries[9]:boundaries[10]])))
        out_terminal_flag = self.final_proj_terminal_flag(self.activation(self.terminal_flag_layer_norm(repeated_residual_output[:, boundaries[10]:boundaries[11]])))

        out_state = torch.cat([out_state_obj, out_state_obst, out_state_goal, out_state_prop], dim=-1)
        out_next_state = torch.cat([out_ns_obj, out_ns_obst, out_ns_goal, out_ns_prop], dim=-1)
        output = torch.cat([out_state, out_action, out_reward, out_next_state, out_terminal_flag], dim=-1)
        return output

# Top-Level Module (CompositionalResidualMLPDenoiser)
# Note: We update the observation dimension to remove the one-hot part.
@gin.configurable
class CompositionalResidualMLPDenoiser(nn.Module):
    def __init__(
        self,
        projection_factor, 
        d_in: int,
        dim_t: int = 164,
        num_layers: int = 6,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = True,
        learned_sinusoidal_dim: int = 16,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.projection_factor = projection_factor
        obs_dim = 77 * self.projection_factor
        observation_shape = (obs_dim,)
        self.residual_mlp = CompositionalResidualMLP(
            projection_factor=self.projection_factor,
            observation_shape=observation_shape,
            depth=num_layers
        )
        if cond_dim is not None:
            #self.proj = nn.Linear(d_in + cond_dim, dim_t)
            self.cond_proj = nn.Linear(16, dim_t)
            self.conditional = True  # update as needed
        else:
            #self.cond_proj = nn.Linear(d_in, dim_t)
            self.conditional = False

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    ####################THIS IS THE ORIGINAL COMPOSITIONAL DIFFUSION FORWARD#######################333
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond=None) -> torch.Tensor:
        time_embed = self.time_mlp(timesteps)
        x = x + time_embed
        if self.conditional:
            assert cond is not None
            #print("COND ShAPE: ", cond.shape)
            #################################MODIFIED TO INTEGRATE COND INTO PRESERVE THE COND INFORMATION FOR RECONSTRUCTION)
            x = x + self.cond_proj(cond)
            #x = torch.cat((x, cond), dim=-1)
        return self.residual_mlp(x)
    
