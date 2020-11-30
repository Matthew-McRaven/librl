import functools

import torch

from .log import *


def cc_episodic_trainer(train_info, task_dist, train_fn, log_fn = cc_reward_logger):
    for epoch in range(train_info['epochs']):
        task_samples = task_dist.sample(train_info['task_count'])
        train_fn(task_samples)
        log_fn(epoch, task_samples)

def cls_trainer(train_info, task_dist, train_fn, log_fn = cls_accuracy_logger):
    for epoch in range(train_info['epochs']):
        task_samples = task_dist.sample(train_info['task_count'])
        result = train_fn(task_samples)
        log_fn(epoch, task_samples, result)