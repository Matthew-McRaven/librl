import functools

import torch
# Sample logging functions.
##############
# CC Loggers #
##############
def cc_reward_logger(epoch, task_samples):
    rewards = len(task_samples) * [None]
    for idx, task in enumerate(task_samples):
        rewards[idx] = sum(task.trajectories[0].reward_buffer.view(-1))

    mean_reward = (sum(rewards)/len(rewards)).item() # type: ignore
    print(f"R^bar_({epoch}) = {mean_reward}.")

# Useful logger for robotics where actions' magitudes really matter.
def cc_action_reward_logger(epoch, task_samples):
    rewards, mu_act = len(task_samples) * [None],  len(task_samples) * [None]
    for idx, task in enumerate(task_samples):
        mu_act[idx] = torch.mean(task.trajectories[0].action_buffer.type(torch.float32), (0)) # type: ignore
        rewards[idx] = sum(task.trajectories[0].reward_buffer.view(-1))

    mean_reward = (sum(rewards)/len(rewards)).item() # type: ignore
    mean_action = functools.reduce(lambda x, y: x+y, mu_act, 0).mean()
    max_action = functools.reduce(lambda x, y: torch.max(x.abs(), y.abs()), mu_act)
    print(f"R^bar_({epoch}) = {mean_reward} with {mean_action:.4f} {max_action.data}.")

#########################
# Classification Loggers#
#########################
# Logger that records per-task accuracy each epoch
def cls_accuracy_logger(epoch, task_samples, correct_total_pairs):
    correct, total = correct_total_pairs
    print(correct, total)
    for idx, _ in enumerate(task_samples):
        print(f"Accuracy on task {idx} of {correct[idx]/total[idx]}")