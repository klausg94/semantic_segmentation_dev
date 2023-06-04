import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union


def get_loss_function(config, num_classes, device):
    if config["loss_function"] == "cross_entropy":
        args = {}
        if not (isinstance(config['class_weights'], list) and len(config['class_weights'])==num_classes):
            pass
        else:
            args["weight"] = torch.tensor(config['class_weights'])
        if "ignore_index" in config.keys() and isinstance(config['ignore_index'], int):
            args["ignore_index"] = config['ignore_index']
        criterion = nn.CrossEntropyLoss(**args).to(device)
    
    if config["loss_function"] == "focal_loss":
        args = {}
        if not (isinstance(config['class_weights'], list) and len(config['class_weights'])==num_classes):
            pass
        else:
            args["weight"] = torch.tensor(config['class_weights'])
        if "ignore_index" in config.keys() and isinstance(config['ignore_index'], int):
            args["ignore_index"] = config['ignore_index']
        args["gamma"] = config["gamma_focal_loss"]
        criterion = FocalLoss(**args).to(device)
    
    return criterion


class FocalLoss(nn.Module):
    # https://github.com/mathiaszinnen/focal_loss_torch
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2
    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weight: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weight is None or isinstance(weight, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weight))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.register_buffer('weight', weight)

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weight is None:
            return torch.ones(target.shape[0])
        weight = target * self.weight
        return weight.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target.flatten(1)
        target = target * (target!=self.ignore_index) 
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        x = nn.functional.softmax(x, dim=1)
        x = x.flatten(2).swapaxes(1,2).contiguous()
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x