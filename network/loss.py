import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)


class StaticLoss(nn.Module):
    def __init__(self, num_classes=19, gamma=1.0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        super(StaticLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = num_classes
        self.size_average = size_average
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight
        self.raw = False
        if (num_classes < 19):
            self.raw = True

    def forward(self, input, target, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C

        if self.raw:
            target_left, target_right, target_up, target_down = target, target, target, target
            target_left[:, :-1, :] = target[:, 1:, :]
            target_right[:, 1:, :] = target[:, :-1, :]
            target_up[:, :, 1:] = target[:, :, :-1]
            target_down[:, :, :-1] = target[:, :, 1:]
            target_left, target_right, target_up, target_down = target_left.view(-1), target_right.view(-1), target_up.view(-1), target_down.view(-1)
            target_left2, target_right2, target_up2, target_down2 = target, target, target, target
            target_left2[:, :-1, 1:] = target[:, 1:, :-1]
            target_right2[:, 1:, 1:] = target[:, :-1, :-1]
            target_up2[:, 1:, :-1] = target[:, :-1, 1:]
            target_down2[:, :-1, :-1] = target[:, 1:, 1:]
            target_left2, target_right2, target_up2, target_down2 = target_left2.view(-1), target_right2.view(-1), target_up2.view(-1), target_down2.view(-1)

        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]
            if self.raw:
                target_left, target_right, target_up, target_down = target_left[valid], target_right[valid], target_up[valid], target_down[valid]
                target_left2, target_right2, target_up2, target_down2 = target_left2[valid], target_right2[valid], target_up2[valid], target_down2[valid]

        if self.one_hot:
            target_onehot = one_hot(target, input.size(1))
            if self.raw:
                target_onehot2 = one_hot(target_left, input.size(1))+one_hot(target_right, input.size(1))\
                                 + one_hot(target_up, input.size(1))+one_hot(target_down, input.size(1)) \
                                 + one_hot(target_left2, input.size(1)) + one_hot(target_right2, input.size(1)) \
                                 + one_hot(target_up2, input.size(1)) + one_hot(target_down2, input.size(1))
                target_onehot = target_onehot+target_onehot2
                target_onehot[target_onehot > 1] = 1

        probs = F.softmax(input, dim=1)
        probs = (self.weights*probs * target_onehot).max(1)[0]
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
