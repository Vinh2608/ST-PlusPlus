import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1, dim=-1)
    # loss_adent = -H.
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def cross_entropy(input, target):
    input = input.softmax(dim=-1)
    return torch.mean(-torch.sum(target * torch.log(input), 1))

def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

def consistency_loss(logits_s, logits_w, p_target, p_model, criterion, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        if use_DA:
            if p_model == None:
                p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

        max_probs, max_idx = torch.max(pseudo_label, dim=-1) # [0,4 0.3 0.3] -> [0.4] va [0]
        mask = max_probs.ge(p_cutoff).float()

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx) * mask
            # print(ce_loss(logits_s, max_idx))

        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = criterion(logits_s, pseudo_label) * mask
        
        #p1, p2 = torch.softmax(logits_s, dim=-1), torch.softmax(logits_w, dim=-1)
        cov_loss = 0 #contras_cls(p1, p2)
        return masked_loss.mean(), mask.mean(), None, max_idx.long(), p_model, cov_loss

    else:
        assert Exception('Not Implemented consistency_loss')

@torch.no_grad()
def masking(logits_x_ulb, threshold, use_hard_label=True, T=0.5):    

    if use_hard_label:
        pseudo_label = torch.argmax(logits_x_ulb, dim=-1).detach()
    else:
        pseudo_label = torch.softmax(logits_x_ulb / T, dim=-1).detach()
        #pseudo_label = algorithm.compute_prob(logits_x_ulb.detach() / algorithm.T)
    loss_w = ce_loss(logits_x_ulb, pseudo_label, reduction='none')
    mask = loss_w.le(threshold).to(logits_x_ulb.dtype)
    return mask

def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist

@torch.no_grad()
def gen_ulb_targets(logits, 
                    use_hard_label=True, 
                    T=1.0,
                    softmax=True, # whether to compute softmax for logits, input must be logits
                    label_smoothing=0.0):
    
    """
    generate pseudo-labels from logits/probs

    Args:
        algorithm: base algorithm
        logits: logits (or probs, need to set softmax to False)
        use_hard_label: flag of using hard labels instead of soft labels
        T: temperature parameters
        softmax: flag of using softmax on logits
        label_smoothing: label_smoothing parameter
    """

    logits = logits.detach()
    if use_hard_label:
        # return hard label directly
        pseudo_label = torch.argmax(logits, dim=-1)
        if label_smoothing:
            pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
        return pseudo_label
    
    # return soft label
    if softmax:
        pseudo_label = torch.softmax(logits / T, dim=-1)
        #pseudo_label = algorithm.compute_prob(logits / T)
    else:
        # inputs logits converted to probabilities already
        pseudo_label = logits
    return pseudo_label

def consistency_loss_(logits_s, logits_w, threshold, name='ce'):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    mask = masking(logits_w, threshold)
    targets = gen_ulb_targets(logits_w)
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits_s, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits_s, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()