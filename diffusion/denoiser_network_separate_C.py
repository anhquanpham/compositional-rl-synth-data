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
THIS VERSION IS USING SEPARATE COMP FOR STATE AND NEXT STATE
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
# Optimized _VectorEncoder
##########################
class _VectorEncoder(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self._observation_shape = observation_shape
        hidden_units = default(hidden_units, lambda: [256, 256])
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._feature_size = hidden_units[-1]
        self._activation = activation
        self._use_dense = use_dense

        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        # When dense connections are not needed, pack layers in Sequential
        if not use_dense:
            layers = []
            for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
                fc = nn.Linear(in_unit, out_unit) if not (use_dense and i > 0) else nn.Linear(in_unit + observation_shape[0], out_unit)
                layers.append(fc)
                layers.append(activation)
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_unit))
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
            self.encoder = nn.Sequential(*layers)
        else:
            # For dense connections, we retain the loop for flexibility.
            self._fcs = nn.ModuleList()
            self._bns = nn.ModuleList()
            self._dropouts = nn.ModuleList()
            for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
                if use_dense and i > 0:
                    in_unit += observation_shape[0]
                self._fcs.append(nn.Linear(in_unit, out_unit))
                if use_batch_norm:
                    self._bns.append(nn.BatchNorm1d(out_unit))
                if dropout_rate is not None:
                    self._dropouts.append(nn.Dropout(dropout_rate))

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_dense:
            return self.encoder(x)
        else:
            h = x
            for i, fc in enumerate(self._fcs):
                if i > 0:
                    h = torch.cat([h, x], dim=1)
                h = self._activation(fc(h))
                if self._use_batch_norm:
                    h = self._bns[i](h)
                if self._dropout_rate is not None:
                    h = self._dropouts[i](h)
            return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def last_layer(self) -> nn.Linear:
        if not self._use_dense:
            # Assuming the last layer in Sequential is the final dropout/activation,
            # retrieve the last Linear module manually.
            for layer in reversed(self.encoder):
                if isinstance(layer, nn.Linear):
                    return layer
        else:
            return self._fcs[-1]


##########################
# Optimized Encoder (Abstract Base)
##########################
class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError


##########################
# Optimized CompositionalMLP
##########################
class CompositionalMLP(nn.Module):
    """
    Compositional MLP module with optimized batch processing.
    """
    def __init__(
        self,
        sizes: Sequence[Sequence[int]],
        num_modules: Sequence[int],
        module_assignment_positions: Sequence[int],
        module_inputs: Sequence[str],
        interface_depths: Sequence[int],
        graph_structure: Sequence[Sequence[int]],
        init_w: float = 3e-3,
        hidden_activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = nn.Identity,
        hidden_init: Optional[nn.Module] = fanin_init,
        b_init_value: float = 0.1,
        layer_norm: bool = False,
        layer_norm_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        layer_norm_kwargs = default(layer_norm_kwargs, dict())
        self.sizes = sizes
        self.num_modules = num_modules
        self.module_assignment_positions = module_assignment_positions
        self.module_inputs = module_inputs  # keys in a dict
        self.interface_depths = interface_depths
        self.graph_structure = graph_structure
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm

        self.module_list = nn.ModuleList()
        for graph_depth in range(len(graph_structure)):
            for j in graph_structure[graph_depth]:
                mod_dict = nn.ModuleDict()
                mod_dict["pre_interface"] = nn.ModuleList()
                mod_dict["post_interface"] = nn.ModuleList()
                for k in range(num_modules[j]):
                    layers_pre = []
                    layers_post = []
                    for i in range(len(sizes[j]) - 1):
                        if i == interface_depths[j]:
                            input_size = sum(sizes[j_prev][-1] for j_prev in graph_structure[graph_depth - 1])
                            input_size += sizes[j][i]
                        else:
                            input_size = sizes[j][i]
                        fc = nn.Linear(input_size, sizes[j][i + 1])
                        if graph_depth < len(graph_structure) - 1 or i < len(sizes[j]) - 2:
                            hidden_init(fc.weight)
                            fc.bias.data.fill_(b_init_value)
                            act = hidden_activation
                            layer_norm_this = layer_norm
                        else:
                            fc.weight.data.uniform_(-init_w, init_w)
                            fc.bias.data.uniform_(-init_w, init_w)
                            act = output_activation
                            layer_norm_this = None
                        

                        if layer_norm_this is not None:
                            new_layer = [fc, nn.LayerNorm(sizes[j][i + 1]), act()]
                        else:
                            new_layer = [fc, act()]

                        if i < interface_depths[j]:
                            layers_pre += new_layer
                        else:
                            layers_post += new_layer

                    if layers_pre:
                        mod_dict["pre_interface"].append(nn.Sequential(*layers_pre))
                    else:
                        mod_dict["pre_interface"].append(nn.Identity())
                    mod_dict["post_interface"].append(nn.Sequential(*layers_post))
                self.module_list.append(mod_dict)

    def forward(self, input_val: torch.Tensor, return_preactivations: bool = False) -> torch.Tensor:
        device = input_val.device
        if input_val.ndim > 2:
            input_val = input_val.squeeze(0)
        if return_preactivations:
            raise NotImplementedError("Return pre-activations not implemented.")
        x = None
        for graph_depth in range(len(self.graph_structure)):
            x_post = []
            for j in self.graph_structure[graph_depth]:
                if input_val.ndim == 1:
                    x_pre = input_val[self.module_inputs[j]]
                    onehot = input_val[self.module_assignment_positions[j]]
                    module_index = onehot.nonzero()[0]
                    x_pre = self.module_list[j]["pre_interface"][module_index](x_pre)
                    if x is not None:
                        x_pre = torch.cat((x, x_pre), dim=-1)
                    x_post.append(self.module_list[j]["post_interface"][module_index](x_pre))
                else:
                    x_post_tmp = torch.empty(input_val.shape[0], self.sizes[j][-1]).to(device)
                    x_pre = input_val[:, self.module_inputs[j]]
                    onehot = input_val[:, self.module_assignment_positions[j]]
                    module_indices = onehot.nonzero(as_tuple=True)
                    assert (module_indices[0] == torch.arange(module_indices[0].shape[0]).to(device)).all()
                    module_indices_1 = module_indices[1]
                    for module_idx in range(self.num_modules[j]):
                        mask = module_indices_1 == module_idx
                        mask_to_input_idx = mask.nonzero()
                        x_pre_this = self.module_list[j]["pre_interface"][module_idx](x_pre[mask])
                        if x is not None:
                            x_pre_this = torch.cat((x[mask], x_pre_this), dim=-1)
                        x_post_this = self.module_list[j]["post_interface"][module_idx](x_pre_this)
                        mask_to_input_idx = mask_to_input_idx.expand(mask_to_input_idx.shape[0], x_post_this.shape[1])
                        x_post_tmp.scatter_(0, mask_to_input_idx, x_post_this)
                    x_post.append(x_post_tmp)
            x = torch.cat(x_post, dim=-1)
        return x


##########################
# _CompositionalEncoder and CompositionalEncoder (unchanged)
##########################
class _CompositionalEncoder(_VectorEncoder):
    """_CompositionalEncoder class."""
    def __init__(self, encoder_kwargs: dict, observation_shape: Sequence[int], init_w: float = 3e-3):
        super().__init__(
            observation_shape,
            hidden_units=None,
            use_batch_norm=False,
            dropout_rate=None,
            use_dense=False,
            activation=nn.ReLU(),
        )
        self._observation_shape = observation_shape
        self.encoder_kwargs = encoder_kwargs
        sizes = encoder_kwargs["sizes"]
        output_dim = encoder_kwargs["output_dim"]
        num_modules = encoder_kwargs["num_modules"]
        module_assignment_positions = encoder_kwargs["module_assignment_positions"]
        module_inputs = encoder_kwargs["module_inputs"]
        interface_depths = encoder_kwargs["interface_depths"]
        graph_structure = encoder_kwargs["graph_structure"]
        sizes = list(sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [output_dim]
        self._feature_size = sizes[-1][-1]
        self.comp_mlp = CompositionalMLP(
            sizes=sizes,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure,
            init_w=init_w,
        )

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.comp_mlp(x)

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError("CompositionalEncoder does not have last_layer")

class CompositionalEncoder(_CompositionalEncoder, Encoder):
    """Implements the actual Compositional Encoder."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fc_encode(x)


##########################
# Optimized CompositionalResidualBlock and CompositionalResidualMLP
##########################
class CompositionalResidualBlock(nn.Module):
    def __init__(self, projection_factor, encoder_kwargs, observation_shape, layer_norm):
        super().__init__()
        self.projection_factor = projection_factor
        self.state_compositional_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=observation_shape,
        )
        self.next_state_compositional_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=observation_shape,
        )
        self.ln = nn.LayerNorm(164 * self.projection_factor) if layer_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.mlp_linear = nn.Linear(164 * self.projection_factor, 164 * self.projection_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pf = self.projection_factor
        # Extract batch components
        current_state = x[:, :77 * pf]         # [batch, 77*pf]
        action = x[:, 77 * pf:85 * pf]           # [batch, 8*pf]
        reward = x[:, 85 * pf:86 * pf]           # [batch, 1*pf]
        next_state = x[:, 86 * pf:163 * pf]        # [batch, 77*pf]
        terminal_flag = x[:, 163 * pf:164 * pf]    # [batch, 1*pf]
        onehot = x[:, 164 * pf:164 * pf + 16]       # [batch, 16]

        # Concatenate onehot with current and next states
        current_state_with_onehot = torch.cat([current_state, onehot], dim=-1)
        next_state_with_onehot = torch.cat([next_state, onehot], dim=-1)

        # Process both states in one forward pass via stacking
        #stacked = torch.cat([current_state_with_onehot, next_state_with_onehot], dim=0)
        #stacked_emb = self.compositional_encoder(stacked)

        #current_state_emb, next_state_emb = torch.chunk(stacked_emb, 2, dim=0)

        #TEST WITH SEPARATE FORWARD PASSES FOR STATES AND NEXT STATES
        current_state_emb = self.state_compositional_encoder(current_state_with_onehot)
        next_state_emb = self.next_state_compositional_encoder(next_state_with_onehot)

        learned_residual = torch.cat([current_state_emb, action, reward, next_state_emb, terminal_flag], dim=-1)
        learned_residual = self.mlp_linear(self.activation(self.ln(learned_residual)))
        first_part = x[:, :164 * pf] + learned_residual
        return torch.cat([first_part, onehot], dim=-1)

class CompositionalResidualMLP(nn.Module):
    def __init__(self, projection_factor, encoder_kwargs, observation_shape, depth: int, activation: str = "relu", layer_norm: bool = False):
        super().__init__()
        self.projection_factor = projection_factor
        self.network = nn.Sequential(
            *[CompositionalResidualBlock(self.projection_factor, encoder_kwargs, observation_shape, layer_norm)
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
        # Slice input into components
        current_state = x[:, :77]         # [batch, 77]
        action = x[:, 77:85]              # [batch, 8]
        reward = x[:, 85:86]              # [batch, 1]
        next_state = x[:, 86:163]         # [batch, 77]
        terminal_flag = x[:, 163:164]     # [batch, 1]
        onehot = x[:, 164:180]            # [batch, 16]

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

        projected_input = torch.cat([proj_current_state, proj_action, proj_reward, proj_next_state, proj_terminal_flag, onehot], dim=-1)
        repeated_residual_output = self.network(projected_input)

        # Precomputed boundaries (multiplied by pf)
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


##########################
# Top-Level CompositionalResidualMLPDenoiser
##########################
@gin.configurable
class CompositionalResidualMLPDenoiser(nn.Module):
    def __init__(
        self,
        projection_factor,
        d_in: int,
        dim_t: int = 164,
        num_layers: int = 3,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = True,
        learned_sinusoidal_dim: int = 16,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        reference_observation_positions = {
            'object-state': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]), 
            'obstacle-state': np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 
            'goal-state': np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]),
            'robot0_proprio-state': np.array([45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                                               62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]),
            'object_id': np.array([77, 78, 79, 80]), 
            'robot_id': np.array([81, 82, 83, 84]), 
            'obstacle_id': np.array([85, 86, 87, 88]), 
            'subtask_id': np.array([89, 90, 91, 92])
        }
        self.projection_factor = projection_factor
        observation_positions = adapt_observation_positions(reference_observation_positions, self.projection_factor)
        sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))
        module_names = ["obstacle_id", "object_id", "subtask_id", "robot_id"]
        module_input_names = ["obstacle-state", "object-state", "goal-state", "robot0_proprio-state"]
        module_assignment_positions = [observation_positions[key] for key in module_names]
        interface_depths = [-1, 1, 2, 3]
        graph_structure = [[0], [1], [2], [3]]
        num_modules = [len(observation_positions[key]) for key in module_names]
        module_inputs = []
        for key in module_input_names:
            if isinstance(key, list):
                module_inputs.append(np.concatenate([observation_positions[k] for k in key], axis=0))
            else:
                module_inputs.append(observation_positions[key])
        obs_dim = 77 * self.projection_factor + 16
        act_dim = 8 * self.projection_factor
        output_dim = 77 * self.projection_factor
        observation_shape = (obs_dim,)
        encoder_kwargs = {
            "sizes": sizes,
            "obs_dim": obs_dim,
            "output_dim": output_dim if output_dim is not None else act_dim,
            "num_modules": num_modules,
            "module_assignment_positions": module_assignment_positions,
            "module_inputs": module_inputs,
            "interface_depths": interface_depths,
            "graph_structure": graph_structure,
        }
        self.residual_mlp = CompositionalResidualMLP(
            projection_factor=self.projection_factor,
            encoder_kwargs=encoder_kwargs,
            observation_shape=observation_shape,
            depth=num_layers
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
        time_embed = self.time_mlp(timesteps)
        x = x + time_embed
        if self.conditional:
            assert cond is not None
            x = torch.cat((x, cond), dim=-1)
        return self.residual_mlp(x)


def adapt_observation_positions(reference_observation_positions: dict, projection_factor: int) -> dict:
    adjusted_observation_positions = {}
    current_position = 0
    for key, positions in reference_observation_positions.items():
        if key not in ['object_id', 'robot_id', 'obstacle_id', 'subtask_id']:
            num_components = len(positions)
            new_num_components = num_components * projection_factor
            new_positions = np.arange(current_position, current_position + new_num_components)
            adjusted_observation_positions[key] = new_positions
            current_position += new_num_components
        else:
            num_components = len(positions)
            new_positions = np.arange(current_position, current_position + num_components)
            adjusted_observation_positions[key] = new_positions
            current_position += num_components
    return adjusted_observation_positions

