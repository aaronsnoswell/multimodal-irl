"""A simple multi-modal discrete state and action MDP

In this stochastic grid world environment, an agent must navigate to a single goal
state as fast as possible.
Depending on the current reward mode, the agent must also try and;

 * `dry` mode: Avoid wet squares
 * `wet` mode: Avoid dry squares
 * `any` mode: Ignore wet/dry status of squares (no penalty for touching either)

This MDP was first introduced in Babeş-Vroman et al., "Apprenticeship Learning about
Multiple Intentions", in ICML, 2011
"""


import gym
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding

from mdp_extras import DiscreteExplicitExtras, Disjoint, Linear
from mdp_extras.utils import compute_parents_children


class PuddleWorldEnv(gym.Env):

    # Features of the environment
    FEATURES = {"dry": 0, "wet": 1, "goal": 2}

    # Actions the agent can take
    ACTION_MAP = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
    }

    ACTION_SYMBOLS = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
    }

    # Coordinate system is origin at top left, +Y down, +X right
    ACTION_VECTORS = {
        0: np.array([-1, 0]),
        1: np.array([1, 0]),
        2: np.array([0, -1]),
        3: np.array([0, 1]),
    }

    # Rewards values
    REWARD_VALUES = {"very_bad": -10.0, "bad": -1.0, "meh": 0.0}

    # Different reward modes
    REWARD_MODES = {
        "wet": [REWARD_VALUES["very_bad"], REWARD_VALUES["bad"], REWARD_VALUES["meh"]],
        "dry": [REWARD_VALUES["bad"], REWARD_VALUES["very_bad"], REWARD_VALUES["meh"]],
        "any": [REWARD_VALUES["bad"], REWARD_VALUES["bad"], REWARD_VALUES["meh"]],
    }

    # Probability of a non-goal state being wet
    P_WET = 5.0 / 24.0

    # Probability of a non-goal state being a start state
    P_START = 11.0 / 24.0

    # Gym Env properties
    metadata = {"render.modes": ["human", "rgb_array", "ascii"]}
    reward_range = (min(REWARD_VALUES.values()), max(REWARD_VALUES.values()))

    def __init__(
        self,
        width,
        height,
        *,
        mode="dry",
        wind=0.2,
        reward_values=(-10, -1, 0),
        seed=None
    ):
        """C-tor
        
        Args:
            width (int): Width of PuddleWorld
            height (int): Height of PuddleWorld
            
            mode (str): Reward mode to use, options are 'wet', 'dry', and 'any'
            wind (float): Wind (random action) probability
            reward_values (list): Optional list of very bad, bad and meh reward values
            seed (int): Random seed to use
        """

        assert mode in self.REWARD_MODES.keys(), "Invalid mode"
        self._mode = mode

        assert len(reward_values) == 3, "Invalid parameter reward_values"

        self.REWARD_VALUES["very_bad"] = reward_values[0]
        self.REWARD_VALUES["bad"] = reward_values[1]
        self.REWARD_VALUES["meh"] = reward_values[2]

        self.seed(seed)

        self._width = width
        self._height = height
        self.observation_space = spaces.Discrete(self._width * self._height)
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))

        self._wind = wind

        # Build the feature matrix
        # 1. Select random goal
        goal_state = np.random.choice(np.arange(self.observation_space.n))
        self._feature_matrix = np.zeros((self._height, self._width), dtype=int)
        self._feature_matrix.flat[goal_state] = self.FEATURES["goal"]

        while True:

            # 2. Non-goal states may be wet/dry
            for s, (y, x) in enumerate(
                it.product(range(self._height), range(self._width))
            ):
                if s == goal_state:
                    continue
                self._feature_matrix[y, x] = np.random.rand() <= self.P_WET

            # 3. Non-goal states may be start states
            self._start_states = []
            for s in range(self.observation_space.n):
                if s == goal_state:
                    continue
                if np.random.rand() <= self.P_START:
                    self._start_states.append(s)
            self._start_states = np.array(self._start_states)

            # Check that we have at least one wet and one dry
            # and that we have at least one start state
            if (
                self.FEATURES["wet"] in self._feature_matrix.flat
                and self.FEATURES["dry"] in self._feature_matrix.flat
                and len(self._start_states) > 0
            ):
                break

        # Prepare IExplicitEnv items
        self._states = np.arange(self.observation_space.n, dtype=int)
        self._actions = np.arange(self.action_space.n, dtype=int)

        self._p0s = np.zeros(self.observation_space.n)
        self._p0s[self._start_states] = 1.0
        self._p0s /= np.sum(self._p0s)

        self._terminal_state_mask = np.zeros(self.observation_space.n)
        self._terminal_state_mask[goal_state] = 1.0

        # Compute s, a, s' transition matrix
        self._t_mat = self._build_transition_matrix()

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        self._gamma = 0.99

        # Build linear state reward vector
        self._state_rewards = np.array(
            [
                self.REWARD_MODES[self._mode][self._feature_matrix.flat[s]]
                for s in self._states
            ],
            dtype=float,
        )
        self._state_action_rewards = None
        self._state_action_state_rewards = None

        self.state = self.reset()

    def seed(self, seed=None):
        """Seed the environment"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _build_transition_matrix(self):
        """Assemble the transition matrix from self._feature_matrix"""
        # Compute s, a, s' transition matrix
        transition_matrix = np.zeros(
            (len(self._states), len(self._actions), len(self._states))
        )
        for s1, a in it.product(self._states, self._actions):
            # Convert states to coords, action to vector
            yx1 = np.array(self._s2yx(s1))
            av = PuddleWorldEnv.ACTION_VECTORS[a]

            # If moving out of bounds, return to current state
            if self._oob(yx1 + av):
                transition_matrix[s1, a, s1] = 1.0
            else:
                target_state = self._yx2s(yx1 + av)
                alternate_states = self._nei(s1)
                alternate_states.remove(target_state)

                # Wind might move us to an alternate state
                transition_matrix[s1, a, alternate_states] = self._wind / len(
                    alternate_states
                )

                # Target state gets non-wind probability
                transition_matrix[s1, a, target_state] = 1.0 - self._wind

        # Ensure that goal state(s) are terminal
        goal_states = np.where(
            self._feature_matrix.flatten() == PuddleWorldEnv.FEATURES["goal"]
        )
        transition_matrix[goal_states, :, :] = 0.0

        return transition_matrix

    def reset(self):
        """Reset the environment"""
        self.state = np.random.choice(self._states, p=self._p0s)
        return self.observe()

    def step(self, action):
        """Step the environment"""

        # Verify action
        assert self.action_space.contains(action), "Invalid action: {}".format(action)

        # Apply action
        self.state = np.random.choice(
            self._states, p=self._t_mat[self.state, action, :].flatten(),
        )

        return (
            self.observe(self.state),
            self.reward(self.state),
            self.done(self.state),
            {},
        )

    def _s2yx(self, state):
        """Convert state to (y, x) coordinates"""
        assert self.observation_space.contains(state)
        y = state // self._width
        x = state - y * self._width
        return (y, x)

    def _yx2s(self, yx):
        """Convert y, x tuple to state"""
        y, x = yx
        assert 0 <= y < self._height
        assert 0 <= x < self._width
        return y * self._width + x

    def _oob(self, yx):
        """Check if a y, x coordinate is 'out of bounds'"""
        try:
            return not self.observation_space.contains(self._yx2s(yx))
        except AssertionError:
            return True

    def _nei(self, state=None):
        """Get neighbours of a state"""
        if state is None:
            state = self.state

        y, x = self._s2yx(state)
        neighbours = []
        if y > 0:
            neighbours.append(self._yx2s((y - 1, x)))
        if y < self._height - 1:
            neighbours.append(self._yx2s((y + 1, x)))
        if x > 0:
            neighbours.append(self._yx2s((y, x - 1)))
        if x < self._width - 1:
            neighbours.append(self._yx2s((y, x + 1)))

        return neighbours

    def observe(self, state=None):
        """Get an observation for a state"""
        if state is None:
            state = self.state
        return int(state)

    def reward(self, state):
        """Compute reward given state"""
        if state is None:
            state = self.state
        return self._state_rewards[state]

    def done(self, state):
        """Test if a episode is complete"""
        if state is None:
            state = self.state
        return self._terminal_state_mask[state]

    def render(self, mode="human"):
        """Render the environment"""
        assert mode in self.metadata["render.modes"]

        if mode == "ascii":
            return self._ascii()
        else:
            raise NotImplementedError

    def _ascii(self):
        """Get an ascii string representation of the environment"""
        str_repr = "+" + "-" * self._width + "+\n"
        for row in range(self._height):
            str_repr += "|"
            for col in range(self._width):
                state = self._yx2s((row, col))
                state_feature = self._feature_matrix.flatten()[state]
                if state == self.state:
                    str_repr += "@"
                elif state_feature == self.FEATURES["dry"]:
                    str_repr += " "
                elif state_feature == self.FEATURES["wet"]:
                    str_repr += "#"
                else:
                    str_repr += "G"
            str_repr += "|\n"
        str_repr += "+" + "-" * self._width + "+"
        return str_repr

    def plot_reward(self, r=None, with_text=False):
        """Visualize a given reward function"""

        if r is None:
            r = self._state_rewards

        r = r.reshape((self._height, self._width))
        plt.imshow(
            r, cmap="Reds_r", vmin=self.reward_range[0], vmax=self.reward_range[1]
        )
        if with_text:
            for iy, ix in np.ndindex(r.shape):
                _r_val = r[iy, ix]
                color = "k"
                if _r_val < -5:
                    color = "w"
                plt.text(ix, iy, _r_val, ha="center", va="center", color=color)
        plt.tick_params(
            which="both", bottom=False, labelbottom=False, left=False, labelleft=False
        )


class CanonicalPuddleWorldEnv(PuddleWorldEnv):
    """The canonical puddle world environment"""

    def __init__(self, **kwargs):

        super().__init__(5, 5, **kwargs)

        # Specify the canonical feature matrix
        self._feature_matrix = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 2],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        # Specify start states
        self._start_states = np.array(
            [0, 1, 2, 5, 6, 10, 15, 16, 20, 21, 22], dtype=np.int64
        )

        goal_state = 14

        # Prepare IExplicitEnv items
        self._states = np.arange(self.observation_space.n, dtype=int)
        self._actions = np.arange(self.action_space.n, dtype=int)

        self._p0s = np.zeros(self.observation_space.n, dtype=float)
        self._p0s[self._start_states] = 1.0
        self._p0s /= np.sum(self._p0s)

        self._terminal_state_mask = np.zeros(self.observation_space.n)
        self._terminal_state_mask[goal_state] = 1.0

        # Compute s, a, s' transition matrix
        self._t_mat = self._build_transition_matrix()

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        self._gamma = 0.99

        # Build linear state reward vector
        self._state_rewards = np.array(
            [
                self.REWARD_MODES[self._mode][self._feature_matrix.flat[s]]
                for s in self._states
            ],
            dtype=float,
        )
        self._state_action_rewards = None
        self._state_action_state_rewards = None

        self.state = self.reset()


class SmallPuddleWorldEnv(PuddleWorldEnv):
    """A small puddle world environment for debugging"""

    def __init__(self, **kwargs):

        super().__init__(3, 3, **kwargs)

        # Specify the canonical feature matrix
        self._feature_matrix = np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0],])

        # Specify start states
        self._start_states = np.array([0, 3, 6], dtype=np.int64)

        goal_state = 5

        # Prepare IExplicitEnv items
        self._states = np.arange(self.observation_space.n, dtype=int)
        self._actions = np.arange(self.action_space.n, dtype=int)

        self._p0s = np.zeros(self.observation_space.n)
        self._p0s[self._start_states] = 1.0
        self._p0s /= np.sum(self._p0s)

        self._terminal_state_mask = np.zeros(self.observation_space.n)
        self._terminal_state_mask[goal_state] = 1.0

        # Compute s, a, s' transition matrix
        self._t_mat = self._build_transition_matrix()

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        self._gamma = 0.99

        # Build linear state reward vector
        self._state_rewards = np.array(
            [
                self.REWARD_MODES[self._mode][self._feature_matrix.flat[s]]
                for s in self._states
            ]
        )
        self._state_action_rewards = None
        self._state_action_state_rewards = None

        self.state = self.reset()


def puddle_world_extras(env):
    """Get multi-modal MDP extras for a PuddleWorld env
    
    Args:
        env (PuddleWorldEnv): Environment to build extras from
        
    Returns:
        (DiscreteExplicitExtras): PuddleWorld extras object
        (Disjoint): PuddleWorld feature function
        (dict): Dictionary of (str):(Linear) PuddleWorld reward functions, one for
            each mode
    """
    xtr = DiscreteExplicitExtras(
        env._states,
        env._actions,
        env._p0s,
        env._t_mat,
        env._terminal_state_mask,
        env._gamma,
    )

    phi = Disjoint(Disjoint.Type.OBSERVATION, xtr, env._feature_matrix.flatten())

    rewards = {
        reward_name: Linear(reward_weights)
        for reward_name, reward_weights in env.REWARD_MODES.items()
    }

    return xtr, phi, rewards
