import functools

import torch

def cc_episodic_trainer(train_info, task_dist, train_fn):
    for epoch in range(train_info['epochs']):
        task_samples = task_dist.sample(train_info['task_count'])
        train_fn(task_samples)

        rewards, mu_act = len(task_samples) * [None],  len(task_samples) * [None]
        for idx, task in enumerate(task_samples):
            mu_act[idx] = torch.mean(task.trajectories[0].action_buffer.type(torch.float32), (0)) # type: ignore
            rewards[idx] = sum(task.trajectories[0].reward_buffer.view(-1))

        mean_reward = (sum(rewards)/len(rewards)).item() # type: ignore
        mean_action = functools.reduce(lambda x, y: x+y, mu_act, 0).mean()
        max_action = functools.reduce(lambda x, y: torch.max(x.abs(), y.abs()), mu_act)
        print(f"R^bar_({epoch}) = {mean_reward} with {mean_action:.4f} {max_action.data}.")

def cls_trainer(train_info, task_dist, train_fn):
    for epoch in range(train_info['epochs']):
        task_samples = task_dist.sample(train_info['task_count'])
        train_fn(task_samples)