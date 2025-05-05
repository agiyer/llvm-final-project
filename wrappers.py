
import os
import gym
import imageio
import numpy as np

#class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
# tweaked from stablebaseline3 atari wrappers
class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info
    
# same as JoypadSpace with reset method fixed ; nes_py.wrappers 
class JoypadSpace(gym.Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    # a mapping of buttons to binary values
    _button_map = {
        'right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOOP':   0b00000000,
    }

    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        return list(cls._button_map.keys())

    def __init__(self, env, actions: list):
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(actions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = ' '.join(button_list)

    def step(self, action):
        """
        Take a step using the given action.

        Args:
            action (int): the discrete action to perform

        Returns:
            a tuple of:
            - (numpy.ndarray) the state as a result of the action
            - (float) the reward achieved by taking the action
            - (bool) a flag denoting whether the episode has ended
            - (dict) a dictionary of extra information

        """
        # take the step and record the output
        return self.env.step(self._action_map[action])

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation."""
        return self.env.reset(**kwargs)

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        # get the old mapping of keys to actions
        old_keys_to_action = self.env.unwrapped.get_keys_to_action()
        # invert the keys to action mapping to lookup key combos by action
        action_to_keys = {v: k for k, v in old_keys_to_action.items()}
        # create a new mapping of keys to actions
        keys_to_action = {}
        # iterate over the actions and their byte values in this mapper
        for action, byte in self._action_map.items():
            # get the keys to press for the action
            keys = action_to_keys[byte]
            # set the keys value in the dictionary to the current discrete act
            keys_to_action[keys] = action

        return keys_to_action

    def get_action_meanings(self):
        """Return a list of actions meanings."""
        actions = sorted(self._action_meanings.keys())
        return [self._action_meanings[action] for action in actions]

# custom video recording wrapper (gym's default was seg faulting)
class VideoRecorder(gym.Wrapper) : 

    # should_record_episode takes in an episode # and outputs boolean of whether to record episode
    def __init__(self, env, video_folder, should_record_episode, fps = 30) : 
        super().__init__(env)

        self.video_folder = video_folder
        os.makedirs(video_folder, exist_ok=True)
        
        self.should_record_episode = should_record_episode
        self.fps = fps
        
        self.episode_id = 0

        self.writer = None

    def reset(self, **kwargs) : 

        print(f'Resetting video. Episode ID: {self.episode_id}')

        obs = self.env.reset(**kwargs)

        # if not recording check if we should start recording
        if self.writer is None and self.should_record_episode(self.episode_id) : 
            
            # start recording
            video_path = os.path.join(self.video_folder, f'episode_{self.episode_id}.mp4')
            self.writer = imageio.get_writer(video_path, fps = self.fps)  

        if self.writer is not None: 

            frame = self.env.render()
            self.writer.append_data(frame)

        return obs

    def step(self, action) : 

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.writer is not None : 

            frame = self.env.render()
            self.writer.append_data(frame)

            if terminated or truncated : 
                self.writer.close()
                self.writer = None

        if terminated or truncated : 
            self.episode_id += 1

        return obs, reward, terminated, truncated, info

    def close(self) : 
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        return self.env.close()
