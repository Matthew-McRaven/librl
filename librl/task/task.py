import gym
import torch

import librl.replay
import librl.task

# Describes a family of related tasks
# Single task object is not shared between episodes within an epoch.
# Otherwise, replay buffers will overwrite each other.
class Task():
    class Definition:
        def __init__(self, ctor, **kwargs):
            self.task_ctor = ctor
            self.task_kwargs = kwargs
        def instance(self):
            return self.task_ctor(**self.task_kwargs)
    def __init__(self, problem_type):
        self._problem_type=problem_type
        
    # Make the problem type effectively constant
    @property
    def problem_type(self):
        return self._problem_type

class ContinuousControlTask(Task):
    # Methods to fill a task's replay buffer
    @staticmethod
    def sample_trajectories(task):
        task.clear_trajectories()
        task.init_env()
        for i in range(task.trajectory_count):
            state = torch.tensor(task.env.reset()).to(task.agent.hypers['device'])
            episode = librl.replay.Episode(task.env.observation_space, task.env.action_space, task.agent.hypers['episode_length'])
            episode.log_done(task.agent.hypers['episode_length'] + 1)
            for t in range(task.agent.hypers['episode_length']):
                
                episode.log_state(t, state)

                action, logprob_action = task.agent.act(state)
                episode.log_action(t, action, logprob_action)
                if task.agent.policy_based: episode.log_policy(t, task.agent.policy_latest)
                x = action.view(-1).detach().cpu().numpy()
                state, reward, done, _ = task.env.step(x)
                state, reward = torch.tensor(state).to(task.agent.hypers['device']), torch.tensor(reward).to(task.agent.hypers['device'])

                episode.log_rewards(t, reward)
                if done: 
                    episode.log_done(t+1)
                    break

            task.add_trajectory(episode)

    def __init__(self, sample_fn = None, env=None, agent=None, trajectories=1):
        super(ContinuousControlTask,self).__init__(librl.task.ProblemTypes.ContinuousControl)
        assert env is not None and agent is not None
        assert isinstance(env, gym.Env)
        if sample_fn is None: sample_fn=ContinuousControlTask.sample_trajectories
        self.env = env
        self.agent = agent
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