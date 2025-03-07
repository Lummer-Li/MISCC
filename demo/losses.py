import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_samples_contrastive_loss(x1, x2, y1, y2, temperature=0.5, base_temperature=0.5):
    """
    Compute centralized clustering alignment loss for cross-modal feature learning

    Parameters
    ----------
    x1: torch.Tensor
        The embedding features of 1-th view, whose shape is [batch_size, feat_dim]
    x2: torch.Tensor
        The embedding features of 2-th view, whose shape is [batch_size, feat_dim]
    y1: torch.Tensor
        The pseudo labels of 1-th view, whose shape is [batch_size]
    y2: torch.Tensor
        The pseudo labels of 2-th view, whose shape is [batch_size]
    temperature: float optional(default 0.5)
        Temperature coefficient to adjust the sharpness of the similarity distribution 
    base_temperature: float optional(default 0.5)
        Basic temperature coefficient to balance loss scales

    Returns 
    -------
    loss: torch.Tensor
        Final loss values
    """
    # To get device
    device = x1.device

    # Calculate similarity
    anchor_dot_contrast = torch.div(torch.matmul(x1, x2.T), temperature)

    # To stable numbers
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  
    logits = anchor_dot_contrast - logits_max.detach() 

    # Create mask
    y1, y2 = y1.view(-1, 1), y2.view(-1, 1)  
    mask = torch.eq(y1, y2.T).float().to(device)  

    # Calculate cross-entropy
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  
    mean_log_prob_pos = (mask * log_prob).sum(1) / torch.abs(mask).sum(1)      

    # Final loss 
    loss = - (temperature / base_temperature) * mean_log_prob_pos  
    return loss.mean()