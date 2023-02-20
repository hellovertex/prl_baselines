from typing import (
    Optional,
    Sequence,
    Type,
    Union,
)

import numpy as np
import torch
from tianshou.utils.net.common import MLP, ModuleType
from torch import nn


class MDL(MLP):
    def __init__(self, input_dim: int,
                 output_dim: int = 0,
                 hidden_sizes: Sequence[int] = (),
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
                 device: Optional[Union[str, int, torch.device]] = None,
                 linear_layer: Type[nn.Linear] = nn.Linear,
                 flatten_input: bool = True, ):
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         hidden_sizes=hidden_sizes,
                         norm_layer=norm_layer,
                         activation=activation,
                         device=device,
                         linear_layer=linear_layer,
                         flatten_input=flatten_input)
        self.sigmoid = nn.Sigmoid()

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        x = self.model(obs)
        return self.sigmoid(x)


def get_model_predict_fold_binary(traindata, hidden_dims, device, merge_labels567=False):
    # network

    net = MDL(input_dim=569,
              output_dim=1,
              hidden_sizes=hidden_dims,
              norm_layer=None,
              activation=nn.ReLU,
              device=device,
              linear_layer=nn.Linear,
              flatten_input=False)
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    return net
