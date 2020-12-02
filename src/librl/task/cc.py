import gym
import numpy as np
import torch

import librl.replay
# The current module will be imported before librl.task is finished, so we
# will need to refer to our parent class by relative path.
from .base import Task as _Task

class ContinuousControlTask(_Task):
    # Aliase batch size as the episode count.
    episode_count=property(_Task.batch_size, _Task.batch_size)

    # Methods to fill a task's sequential replay buffer.
    @staticmethod
    def sample_trajectories(task):
        task.clear_trajectories()
        task.init_env()

        # Massage something that looks like a tensor, ndarray to a tensor on the correct device.
        # If given an iterable of those things, recurssively attempt to massage them into tensors.
        def torchize(maybe_tensor):
            # Sometimes an object is a torch tensor that just needs to be moved to the right device
            if torch.is_tensor(maybe_tensor): return maybe_tensor.to(task.device)
            # Sometimes it is an ndarray
            elif isinstance(maybe_tensor, np.ndarray): return torch.tensor(maybe_tensor).to(task.device) # type: ignore
            # Maybe a simple ierable of things we need to move to torch, like a cartesian product of ndarrays.
            elif isinstance(maybe_tensor, (list, tuple)): return [torchize(item) for item in maybe_tensor]
            else: raise NotImplementedError(f"Don't understand the datatype of {type(maybe_tensor)}")

        for i in range(task.trajectory_count):
            state = torchize(task.env.reset()) # type: ignore
            episode = task.replay_ctor(task.env.observation_space, task.env.action_space, task.episode_length)
            episode.log_done(task.episode_length + 1)
            for t in range(task.episode_length):
                
                episode.log_state(t, state)

                action, logprob_action = task.agent.act(state)
                episode.log_action(t, action, logprob_action)
                if task.agent.policy_based: episode.log_policy(t, task.agent.policy_latest)
                # If our action is a tensor, it needs to be migrated to the CPU for the simulator.
                if torch.is_tensor(action): action = action.view(-1).detach().cpu().numpy()
                state, reward, done, extra_info = task.env.step(action)
                
                if task.agent.allow_callback: task.agent.act_callback(state=state, reward=reward)
                episode.log_extra_info(t, extra_info)

                state = torchize(state)
                # Don't copy reward in to tensor if it already is one; pytorch gets mad.
                if torch.is_tensor(reward): reward = reward.to(task.device) # type: ignore
                else: reward = torch.tensor(reward).to(task.device) # type: ignore

                episode.log_rewards(t, reward)
                if done: 
                    episode.log_done(t+1)
                    break

            task.add_trajectory(episode)

    # Use sample_trajectories as the default sample, unless otherwise specified.
    def __init__(self, sample_fn = sample_trajectories.__func__, replay_ctor=librl.replay.episodic.BoxEpisode, 
    env=None, agent=None, trajectories=1, episode_length=100):

        super(ContinuousControlTask,self).__init__(librl.task.ProblemTypes.ContinuousControl)
        assert env is not None and agent is not None
        assert isinstance(env, gym.Env)

        self.env = env
        self.agent = agent
        self._episode_length = episode_length
        self.trajectory_count = trajectories

        self.sample = sample_fn
        self.replay_ctor = replay_ctor
        self.trajectories = []

    # Override in subclass!!
    def init_env(self):
        raise NotImplemented("Please implement this method in your subclass.")

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
    def clear_trajectories(self):
        self.trajectories = []

    # Let the task control the sampled batch size
    @property
    def episode_length(self):
        return self._episode_length
    @episode_length.setter
    def episode_length(self, value):
        self._episode_length = value

# Example class that allows us to run gym tasks with no extra effort.
class ContinuousGymTask(ContinuousControlTask):
    def __init__(self, **kwargs):
        super(ContinuousGymTask, self).__init__(**kwargs)
    def init_env(self):
        # Most gym environment need no extra init'ing
        pass