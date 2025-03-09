import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import logging 





def feat_loss_multiscale_fn(
    clean_activations, adv_activations, criterion, SOFTMAX_2D, loss_channel
):

    loss = torch.tensor(0.0).cuda()
    softmax = nn.Softmax2d()
    softmax1D = nn.Softmax(dim=-1)
    alpha = 1.0

    for key in clean_activations.keys():
        if "feat" in key:
            B = clean_activations[key].shape[0]

            clean = clean_activations[key]
            adv = adv_activations[key]
            target = None

            clean = clean[:, cfg.trainer.loss.loss_channel]
            adv = adv[:, cfg.trainer.loss.loss_channel]

            if isinstance(criterion, torch.nn.MSELoss):
                loss_layer = criterion(clean, adv)

            else:
                raise NotImplementedError()

            loss += loss_layer

    return loss
