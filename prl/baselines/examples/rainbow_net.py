from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union, List

import numpy as np
import torch
from tianshou.utils.net.common import ModuleType, miniblock
from torch import nn

from tianshou.utils.net.discrete import NoisyLinear


def layer_init(
        layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def scale_obs(module: Type[nn.Module], denom: float = 255.0) -> Type[nn.Module]:
    class scaled_module(module):

        def forward(
                self,
                obs: Union[np.ndarray, torch.Tensor],
                state: Optional[Any] = None,
                info: Dict[str, Any] = {}
        ) -> Tuple[torch.Tensor, Any]:
            return super().forward(obs / denom, state, info)

    return scaled_module


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int = 0,
            hidden_sizes: Sequence[int] = (),
            norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
            activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
            device: Optional[Union[str, int, torch.device]] = None,
            linear_layer: Type[nn.Linear] = nn.Linear,
            flatten_input: bool = True,
    ) -> None:
        super().__init__()
        # monkey-patched because dqn only computes features not q values for rainbow case
        hidden_sizes = [512]
        output_dim = 512
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(
                hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, activation_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = 512
        self.net = nn.Sequential(*model)
        self.flatten_input = flatten_input

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


# class C51(DQN):
#     """Reference: A distributional perspective on reinforcement learning.
#
#     For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """
#
#     def __init__(
#         self,
#         c: int,
#         h: int,
#         w: int,
#         action_shape: Sequence[int],
#         num_atoms: int = 51,
#         device: Union[str, int, torch.device] = "cpu",
#     ) -> None:
#         self.action_num = np.prod(action_shape)
#         super().__init__(c, h, w, [self.action_num * num_atoms], device)
#         self.num_atoms = num_atoms
#
#     def forward(
#         self,
#         obs: Union[np.ndarray, torch.Tensor],
#         state: Optional[Any] = None,
#         info: Dict[str, Any] = {},
#     ) -> Tuple[torch.Tensor, Any]:
#         r"""Mapping: x -> Z(x, \*)."""
#         obs, state = super().forward(obs)
#         obs = obs.view(-1, self.num_atoms).softmax(dim=-1)
#         obs = obs.view(-1, self.action_num, self.num_atoms)
#         return obs, state


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_sizes: List[int],
            norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
            activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
            device: Union[str, int, torch.device] = "cpu",
            linear_layer: Type[nn.Linear] = nn.Linear,
            flatten_input: bool = True,
            num_atoms: int = 51,
            noisy_std: float = 0.5,
            is_dueling: bool = True,
            is_noisy: bool = True,
    ) -> None:
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         hidden_sizes=hidden_sizes,
                         norm_layer=norm_layer,
                         activation=activation,
                         device=device,
                         linear_layer=linear_layer,
                         flatten_input=flatten_input)
        self.action_num = output_dim
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            else:
                return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512), nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms)
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512), nn.ReLU(inplace=True),
                linear(512, self.num_atoms)
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state
