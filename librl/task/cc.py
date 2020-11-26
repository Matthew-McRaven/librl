import gym
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
        for i in range(task.trajectory_count):
            state = torch.tensor(task.env.reset()).to(task.device) # type: ignore
            episode = librl.replay.Episode(task.env.observation_space, task.env.action_space, task.episode_length)
            episode.log_done(task.episode_length + 1)
            for t in range(task.episode_length):
                
                episode.log_state(t, state)

                action, logprob_action = task.agent.act(state)
                episode.log_action(t, action, logprob_action)
                if task.agent.policy_based: episode.log_policy(t, task.agent.policy_latest)
                x = action.view(-1).detach().cpu().numpy()
                state, reward, done, _ = task.env.step(x)
                if task.agent.allow_callback: task.agent.act_callback(state=state, reward=reward)
                state, reward = torch.tensor(state).to(task.device), torch.tensor(reward).to(task.device) # type: ignore

                episode.log_rewards(t, reward)
                if done: 
                    episode.log_done(t+1)
                    break

            task.add_trajectory(episode)

    # Use sample_trajectories as the default sample, unless otherwise specified.
    def __init__(self, sample_fn = sample_trajectories.__func__, env=None, agent=None, trajectories=1, episode_length=100):

        super(ContinuousControlTask,self).__init__(librl.task.ProblemTypes.ContinuousControl)
        assert env is not None and agent is not None
        assert isinstance(env, gym.Env)

        self.env = env
        self.agent = agent
        self._episode_length = episode_length
        self.trajectory_count = trajectories

        self.sample = sample_fn
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