import math

import torch
import torch.optim as optim

def get_optimizer(cfg, net):

    if cfg.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=cfg.lr)
    return optimizer

def get_scheduler(cfg, optimizer):

    if cfg.scheduler == 'cosinewarmup':
        warm_up_epochs = 5
        warm_up_with_cosine_lr = lambda epoch: min(1., (epoch + 1) / warm_up_epochs) if epoch <= warm_up_epochs else 0.5 * (
                math.cos((epoch - warm_up_epochs) / (cfg.epoch - warm_up_epochs) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    elif cfg.scheduler == 'warmup':
        warm_up_epochs = 5
        warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else cfg.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    elif cfg.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1e-1, last_epoch=- 1)

    else:
        lr_scheduler = None

    return lr_scheduler