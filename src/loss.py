import torch
import torch.nn.functional as F
import numpy as np


def hybrid_evidential_ordinal_loss(alpha, targets, num_classes=6,
                                   lambda_reg=0.01, ordinal_weight=0.5,
                                   mse_weight=1.0, annealing_coef=1.0):
    batch_size = alpha.size(0)
    S = alpha.sum(1, keepdim=True)
    prob = alpha / S

    deg_levels = torch.tensor([0., 20., 40., 60., 80., 100.], device=alpha.device)
    pred_value = (prob * deg_levels).sum(1)
    true_value = targets.float() * 20.0
    loss_mse = F.mse_loss(pred_value, true_value)

    class_indices = torch.arange(num_classes, device=alpha.device).float()
    target_indices = targets.float().unsqueeze(1)
    ordinal_distances_sq = (class_indices - target_indices) ** 2
    ordinal_penalty = (prob * ordinal_distances_sq).sum(1).mean()

    y_onehot = F.one_hot(targets, num_classes).float()
    correct_class_evidence = (alpha * y_onehot).sum(1)
    evidence_concentration = -torch.log(correct_class_evidence + 1.0).mean()

    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    kl_div = torch.lgamma(S) - torch.lgamma(alpha_tilde).sum(1) + \
             ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S))).sum(1)
    loss_kl = kl_div.mean() * annealing_coef

    total_loss = mse_weight * loss_mse + \
                 ordinal_weight * ordinal_penalty + \
                 0.2 * evidence_concentration + \
                 lambda_reg * loss_kl

    return total_loss, {
        'loss_mse': loss_mse.item(),
        'ordinal_penalty': ordinal_penalty.item(),
        'evidence_concentration': evidence_concentration.item(),
        'kl_div': loss_kl.item()
    }

def predict_with_uncertainty(alpha, deg_levels=[0, 20, 40, 60, 80, 100]):
    S = alpha.sum(1, keepdim=True)
    prob = alpha / S
    pred_class = prob.argmax(1)

    deg_levels_tensor = torch.tensor(deg_levels, device=alpha.device, dtype=torch.float32)
    pred_value = (prob * deg_levels_tensor).sum(1)

    return {
        'pred_class': pred_class,
        'pred_value': pred_value,
        'prob': prob,
        'total_evidence': S.squeeze()
    }

