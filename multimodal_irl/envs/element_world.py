import gym
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding

from mdp_extras import (
    DiscreteExplicitExtras,
    Disjoint,
    Linear,
    OptimalPolicy,
    nonoverlapping_shared_subsequences,
)
from mdp_extras.utils import compute_parents_children
from unimodal_irl.sw_maxent_irl import maxent_ml_path


class ElementWorldEnv(gym.Env):

    # Rewards values
    REWARD_VALUES = {"very_bad": -10.0, "bad": -5.0, "meh": 0.0}

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

    # Gym Env properties
    metadata = {"render.modes": ["ascii", "human"]}
    reward_range = (min(REWARD_VALUES.values()), max(REWARD_VALUES.values()))

    def __init__(
        self,
        width=5,
        height=5,
        num_elements=5,
        target_element=0,
        wind=0.2,
        gamma=0.935,
        seed=None,
    ):
        """C-tor
        
        Args:
            width (int): Width of PuddleWorld
            height (int): Height of PuddleWorld
            
            wind (float): Wind (random action) probability
            seed (int): Random seed to use
        """
        self.seed(seed)

        self._width = width
        self._height = height
        self._num_elements = num_elements
        self.observation_space = spaces.Discrete(self._width * self._height)
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))

        self._wind = wind

        # Populate feature dictionary
        # Goal feature is signified by '_'
        self.FEATURES = {"_": 0}
        for e in range(1, self._num_elements + 1):
            self.FEATURES.update({chr(65 + e - 1): e})
        self.REV_FEATURES = {v: k for k, v in self.FEATURES.items()}

        # Goal is always the bottom right cell
        goal_state = self.observation_space.n - 1

        # Start is always the top left cell
        self._start_states = np.array([0])

        # Build the feature matrix
        self._feature_matrix = np.zeros((self._height, self._width), dtype=int)
        self._feature_matrix.flat[goal_state] = self.FEATURES["_"]

        while True:

            # 2. Non-goal states get an element
            for s, (y, x) in enumerate(
                it.product(range(self._height), range(self._width))
            ):
                if s == goal_state:
                    continue
                self._feature_matrix[y, x] = np.random.randint(
                    1, self._num_elements + 1
                )

            # Check that all features are represented
            if len(set(self._feature_matrix.flat)) == len(self.FEATURES):
                break
            # Otherwise, try again

        # Prepare other items
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

        self.set_target(target_element)
        self.state = self.reset()

        self._gamma = gamma

    def set_target(self, target_element):
        """Sets the target element, updating the state reward function in the process"""
        self._target_element = target_element
        self._feat2reward = [
            self.REWARD_VALUES["meh"]
            if v == 0
            else (
                self.REWARD_VALUES["bad"]
                if v == self._target_element + 1
                else self.REWARD_VALUES["very_bad"]
            )
            for k, v in self.FEATURES.items()
        ]
        self._state_rewards = np.array(
            [self._feat2reward[self._feature_matrix.flat[s]] for s in self._states]
        )
        self._state_action_rewards = None
        self._state_action_state_rewards = None
        return

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
            av = self.ACTION_VECTORS[a]

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
        goal_states = np.where(self._feature_matrix.flatten() == self.FEATURES["_"])
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

        if mode == "ascii" or mode == "human":
            return self._ascii()
        else:
            raise NotImplementedError

    def _ascii(self, draw_current_state=True):
        """Get an ascii string representation of the environment"""
        str_repr = ""
        s = 0
        for row in self._feature_matrix:
            for f in row:
                if s == self.state and draw_current_state:
                    str_repr += "@"
                else:
                    str_repr += self.REV_FEATURES[f]
                s += 1
            str_repr += "\n"
        return str_repr

    def plot_reward(self, r=None, with_text=False):
        """Visualize a given reward function"""

        raise NotImplementedError()


def element_world_extras(env):
    """Get multi-modal MDP extras for an ElementWorld env
    
    Args:
        env (ElementWorld): Environment to build extras from
        
    Returns:
        (DiscreteExplicitExtras): ElementWorld extras object
        (Disjoint): ElementWorld feature function
        (list): List of (Linear) ElementWorld reward functions, one for each mode
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

    rewards = []
    for target in range(1, env._num_elements + 1):
        r = np.zeros(env._num_elements + 1) + env.REWARD_VALUES["very_bad"]
        r[target] = env.REWARD_VALUES["bad"]
        r[0] = env.REWARD_VALUES["meh"]
        rewards.append(Linear(r))

    return xtr, phi, rewards


def element_world_maxent_mixture_ml_path(xtr, phi, demos, mixture_weights, rewards):
    """Find the ML path for an ElementWorld MaxEnt mixture
    
    Args:
        xtr (DiscreteEplicitExtras): MDP extras
        phi (FeatureFunction): Feature function
        demos (list): List of demonstration paths
        mixture_weights (numpy array): Mixture component weights
        rewards (list): List of Linear rewards
    
    Returns:
        (list): List of ML paths from the mixture - one for each dmeo paths' start and
            end state
    """

    # Get ML paths from MaxEnt mixture
    mixture_paths = []
    mixture_path_lls = []
    for weight, reward in zip(mixture_weights, rewards):
        mode_ml_paths = []
        mode_ml_path_lls = []

        # Shortcut - if all paths share a start and end state, don't re-calculate the ML path N times
        start_states = list(set([demo[0][0] for demo in demos]))
        end_states = list(set([demo[-1][0] for demo in demos]))
        if len(start_states) == len(end_states) == 1:
            # Solve for the ML path once only
            s1 = start_states[0]
            sg = end_states[0]
            ml_path, ml_path_ll = maxent_ml_path(
                xtr, phi, reward, s1, sg, len(demos[0]), with_ll=True
            )
            mode_ml_paths = [ml_path for _ in range(len(demos))]
            mode_ml_path_lls = [ml_path_ll for _ in range(len(demos))]
        else:
            for path in demos:
                s1 = path[0][0]
                sg = path[-1][0]
                ml_path, ml_path_ll = maxent_ml_path(
                    xtr, phi, reward, s1, sg, len(path), with_ll=True
                )
                mode_ml_paths.append(ml_path)
                mode_ml_path_lls.append(ml_path_ll)

        # Add the log weight for this mixture component
        mode_ml_path_lls = np.array(mode_ml_path_lls) + np.log(weight)

        mixture_paths.append(mode_ml_paths)
        mixture_path_lls.append(mode_ml_path_lls)

    # The mixture nominates the max component likelihood path as it's ml path
    ml_path_ids = np.argmax(mixture_path_lls, axis=0)
    mixture_ml_paths = []
    for path_idx in range(len(demos)):
        mixture_idx = ml_path_ids[path_idx]
        mixture_ml_paths.append(mixture_paths[mixture_idx][path_idx])

    return mixture_ml_paths


def percent_distance_missed_metric(path_l, path_gt):
    """Compute % distance missed metric from learned to GT path
    
    Assumes paths are non-cyclic and share start and end states
    """
    gt_path_len = len(path_gt)

    # Find overlapping sub-paths
    shared_state_count = 0
    for shared_subpath in nonoverlapping_shared_subsequences(path_l, path_gt):
        shared_state_count += len(shared_subpath)

    # Compute % shared distance
    pc_shared_distance = shared_state_count / gt_path_len

    # Distance missed is the complement of this
    return 1.0 - pc_shared_distance


def main():
    """Main function"""
    import matplotlib.pyplot as plt

    from pprint import pprint

    import tqdm

    from mdp_extras import q_vi, q2v

    from unimodal_irl.sw_maxent_irl import maxent_ml_path

    num_elements = 4
    size = 10
    env = ElementWorldEnv(
        width=size, height=size, num_elements=num_elements, wind=0.2, gamma=0.99
    )
    xtr, phi, rewards = element_world_extras(env)
    print(env.render())

    ml_path = maxent_ml_path(xtr, phi, rewards[0], 0, len(xtr.states) - 1, size * 3)
    print(ml_path)
    print([s for (s, a) in ml_path])

    # ml_path = maxent_ml_path(xtr, phi, rewards[1], 0, len(xtr.states) - 1, 9)
    # print(ml_path)

    print("Here")

    # fig, axes = plt.subplots(num_elements, 2, sharex=True, sharey=True)
    # plt.set_cmap("Reds_r")
    # for axes_row, reward in zip(axes, rewards):
    #     rs = reward.structured(xtr, phi)[0]
    #     q_star = q_vi(xtr, phi, reward)
    #     v_star = q2v(q_star)
    #     pi_star = OptimalPolicy(q_star)
    #
    #     # s = env.reset()
    #     # done = False
    #     # while not done:
    #     #     a, _ = pi_star.predict(s)
    #     #     s2, r, done, _ = env.step(a)
    #     #     print(env.render())
    #     #     s = s2
    #
    #     demo_star = pi_star.get_rollouts(env, 1000, max_path_length=50)
    #     print(phi.expectation(demo_star, gamma=xtr.gamma))
    #
    #     # plt.sca(axes_row[0])
    #     # plt.imshow(rs.reshape(env._height, env._width), vmin=-10, vmax=0)
    #     # plt.title("Reward")
    #     #
    #     # plt.sca(axes_row[1])
    #     # plt.imshow(v_star.reshape(env._height, env._width))
    #     # plt.title("Value Fn")

    # plt.show()


if __name__ == "__main__":
    main()
