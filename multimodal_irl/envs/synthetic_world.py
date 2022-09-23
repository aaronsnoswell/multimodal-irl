import gym
import random
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from gym import spaces
from gym.utils import seeding

from mdp_extras import *

from unimodal_irl.sw_maxent_irl import maxent_path_logprobs
from scipy.stats import dirichlet


class SyntheticWorldEnv(gym.Env):
    def __init__(
        self,
        num_states=10,
        num_actions=2,
        num_behaviour_modes=3,
        num_feature_dimensions=100,
        discount_factor=0.99,
        seed=None,
    ):
        self.seed(seed)

        if num_behaviour_modes > num_states:
            warnings.warn(
                f"Number of behaviour modes ({num_behaviour_modes}) is greater than number of states ({num_states}) - some modes will overlap in terms of behaviours"
            )

        # Make the states
        self.observation_space = spaces.Discrete(num_states)
        self.state_space = spaces.Discrete(num_states)
        self._states = np.arange(self.observation_space.n, dtype=int)

        # Make the actions
        self.action_space = spaces.Discrete(num_actions)
        self._actions = np.arange(num_actions, dtype=int)

        # Make the initial state distribution by deterministically starting from a random state
        self._start_states = [np.random.choice([k for k in range(len(self._states))])]
        self._p0s = np.zeros(self.observation_space.n)
        self._p0s[self._start_states] = 1.0

        self._gamma = discount_factor
        self._num_behaviour_modes = num_behaviour_modes
        self._current_behaviour_mode = 0
        self._num_feature_dimensions = num_feature_dimensions

        # Construct the behviour feature vectors
        # self._construct_one_hot_behaviour_feature_vectors()
        self._construct_behaviour_feature_vectors()

        # Construct the state feature vectors
        # self._construct_one_hot_state_feature_vectors()
        self._construct_state_feature_vectors()

        # Make transition matrix
        self._t_mat = np.zeros(
            (len(self._states), len(self._actions), len(self._states))
        )

        # Draw samples from unit simplex
        rand_simplex = lambda num_samples: dirichlet.rvs(
            size=num_samples, alpha=np.ones(len(self._states))
        )

        shape = (len(self._states), len(self._actions), len(self._states))
        self._t_mat = rand_simplex(len(self._states) * len(self._actions)).reshape(
            shape
        )

        # Synthetic MDP has no terminal states
        self._terminal_state_mask = np.zeros(self.observation_space.n)

        # We don't need padding - no terminal states
        self._needs_padding = False

        # Reset the MDP, ready for use
        self.state = self.reset()

    def seed(self, seed=None):
        """Set seed"""
        random.seed(seed)
        np.random.seed(seed)

    def render(self, mode="human"):
        """TODO"""
        raise NotImplementedError()

    def set_behaviour_mode(self, mode):
        assert 0 <= mode <= len(self._thetas)
        self._current_behaviour_mode = mode

    def _construct_behaviour_feature_vectors(self):
        self._thetas = np.zeros((1, self._num_feature_dimensions))
        self._thetas[0] = np.array(
            [np.random.choice([-1, 1]) for k in range(self._num_feature_dimensions)]
        )
        for i in range(1, self._num_behaviour_modes):
            theta = np.array(
                [np.random.choice([-1, 1]) for k in range(self._num_feature_dimensions)]
            )

            # Make sure that the new feature vector is sufficiently different to the previous ones
            while self._get_min_pairwise_distance(
                self._thetas, np.array([theta])
            ) < np.sqrt(self._num_feature_dimensions):
                theta = np.array(
                    [
                        np.random.choice([-1, 1])
                        for k in range(self._num_feature_dimensions)
                    ]
                )

            self._thetas = np.append(self._thetas, np.array([theta]), axis=0)

    def _construct_one_hot_behaviour_feature_vectors(self):
        self._thetas = np.zeros(
            (self._num_behaviour_modes, self._num_feature_dimensions)
        )
        random_indices = np.random.choice(
            [k for k in range(self._num_feature_dimensions)],
            size=self._num_behaviour_modes,
        )
        for i in range(self._num_behaviour_modes):
            self._thetas[i][random_indices[i]] = 1.0

    def _construct_state_feature_vectors(self):
        sigma = 0.025
        cov = sigma * sigma * np.identity(self._num_feature_dimensions)
        idx_choice = [k for k in range(self._num_behaviour_modes)]

        stateToBehaviourIdx = [
            (k, np.random.choice(idx_choice)) for k in range(len(self._states))
        ]
        # Dicitionary that maps a mode to a list of states
        indicesByModes = {}
        for mode in range(self._num_behaviour_modes):
            indicesByModes[mode] = [
                stateToBehaviourIdx[k][0]
                for k in range(len(stateToBehaviourIdx))
                if stateToBehaviourIdx[k][1] == mode
            ]

        self._psis = np.zeros((len(self._states), self._num_feature_dimensions))
        for mode in indicesByModes:
            mu = np.array(self._thetas[mode])
            nor = np.random.multivariate_normal(
                mu, cov, size=(len(indicesByModes[mode]),)
            )
            for state_idx in range(len(indicesByModes[mode])):
                self._psis[indicesByModes[mode][state_idx]] = nor[state_idx]

    def _construct_one_hot_state_feature_vectors(self):
        self._psis = np.zeros((len(self._states), self._num_feature_dimensions))
        for i in range(len(self._states)):
            self._psis[i][i] = 1.0

    def _get_min_pairwise_distance(self, a, b):
        return np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).flatten())

    def reset(self):
        self.state = np.random.choice(self._states, p=self._p0s)
        return self.observe()

    def observe(self, state=None):
        """Get an observation for a state"""
        if state is None:
            state = self.state
        return int(state)

    def step(self, action):
        """Step the environment"""

        # Verify action
        assert self.action_space.contains(action), "Invalid action: {}".format(action)

        # Apply action
        self.state = np.random.choice(
            self._states,
            p=self._t_mat[self.state, action, :].flatten(),
        )

        return (
            self.observe(self.state),
            self.reward(self.state),
            self.done(self.state),
            {},
        )

    def reward(self, state):
        """Compute reward given state"""
        if state is None:
            state = self.state

        state_fv = self._psis[state]
        mode_rewards = self._thetas @ state_fv
        return mode_rewards[self._current_behaviour_mode]

    def done(self, state):
        """There are no terminal states in the SyntheticWorld MDP"""
        return False

    def dot_output(self, edge_weight=5.0, view=True):
        """Generate and/or view a DOT visual representation of the MDP

        Args:
            edge_weight (float): How heavy to draw edges
            view (bool): If true, visualize the .dot file immediately
        """

        import graphviz

        dot = graphviz.Digraph("synth-mdp", "Synthetic MDP")
        for s in self._states:
            p0 = self._p0s[s]
            state_rewards = self._thetas @ self._psis[s, :]
            state_rewards = [np.format_float_positional(sr, 2) for sr in state_rewards]
            state_rewards = " ".join(state_rewards)
            dot.node(f"S{s}", f"S{s} ({p0}) +[{state_rewards}]")
        for s1 in range(self._t_mat.shape[0]):
            for a in range(self._t_mat.shape[1]):
                dot.node(f"S{s1}>A{a}", f"S{s1}>A{a}", color="red")
                dot.edge(f"S{s1}", f"S{s1}>A{a}", color="red")
                for s2 in range(self._t_mat.shape[2]):
                    penwidth = self._t_mat[s1, a, s2] * edge_weight
                    penwidth = np.clip(penwidth, a_min=0.01, a_max=None)
                    dot.edge(
                        f"S{s1}>A{a}",
                        f"S{s2}",
                        color="blue",
                        penwidth=f"{penwidth}",
                    )

        dot.render(directory=".", view=view)


class MDPFeature(FeatureFunction):
    def __init__(self, type, xtr, env):
        super().__init__(type)

        self._xtr = xtr
        self._env = env
        self._vec = np.zeros(self._env._num_feature_dimensions)

    def __len__(self):
        return self._env._num_feature_dimensions

    def __call__(self, o1, a=None, o2=None):
        feature_vector = self._vec
        try:
            feature_vector = self._env._psis[o1]
        except IndexError:
            warnings.warn("Got IndexError on MDPFeature.__call__()")
            feature_vector = self._vec
        return feature_vector


def synthetic_world_extras(env):

    xtr = DiscreteExplicitExtras(
        env._states,
        env._actions,
        env._p0s,
        env._t_mat,
        env._terminal_state_mask,
        env._gamma,
    )

    phi = MDPFeature(Disjoint.Type.OBSERVATION, xtr, env)

    # For K behaviour modes, build the list of linear feature functions
    features = np.zeros((env._num_behaviour_modes, env._num_feature_dimensions))
    for i in range(env._num_behaviour_modes):
        features[i] = env._thetas[i]
    rewards = [Linear(features[i]) for i in range(env._num_behaviour_modes)]

    return xtr, phi, rewards


def main():
    """Main function"""

    e = SyntheticWorldEnv(
        num_states=2, num_behaviour_modes=2, num_feature_dimensions=4, seed=41
    )

    print("p0s", e._p0s)
    print()

    print("Behaviour mode feature vectors")
    for th in e._thetas:
        print(th)
    print()

    print("State feature vectors")
    for psi in e._psis:
        print(psi)
    print()

    for a in e._actions:
        print(f"Transition matrix - action A{a}")
        print(e._t_mat[:, a, :])
    print()

    num_rollouts = 10
    max_path_length = 50

    from mdp_extras.soln import vi
    from mdp_extras.soln import OptimalPolicy

    from unimodal_irl.sw_maxent_irl import sw_maxent_irl

    for mode in [0, 1]:

        e.set_behaviour_mode(mode)
        xtr, phi, rewards = synthetic_mdp_extras(e)

        for s in e._states:
            print(f"Mode {mode} - R(S{s})={e.reward(s)}")
        print()

        e.set_behaviour_mode(mode)
        v_s, v_sa = vi(xtr, phi, rewards[mode])
        pistar = OptimalPolicy(v_sa)
        rollouts = pistar.get_rollouts(e, num_rollouts, max_path_length=max_path_length)

        phi_bars = []
        for rollout in rollouts:
            # Compute feature vector for each state in rollouts
            states = np.array(rollout)[:, 0]
            discounted_phis = np.array(
                [e._psis[s] * (e._gamma ** t) for t, s in enumerate(states)]
            )

            # Sum discounted feature vectors
            rollout_phi = np.sum(discounted_phis, axis=0)
            phi_bars.append(rollout_phi)

        # Average phi_bars
        phi_bar_star = np.mean(phi_bars, axis=0)
        print("Phi Bar Star = ")
        print(phi_bar_star)
        print()

        x = e._thetas[mode]

        # nll, grad = sw_maxent_irl(
        #     x,
        #     xtr,
        #     phi,
        #     phi_bar_star,
        #     max_path_length,
        #     nll_only=False
        # )
        #
        # print(nll, grad)

        reward_range = (-1.0, 1.0)
        reward_parameter_bounds = tuple(reward_range for _ in range(len(phi)))
        minimize_options = {}
        minimize_kwargs = {}
        theta0 = np.random.randn(len(phi))

        from scipy.optimize import minimize

        res_lbfgs = minimize(
            sw_maxent_irl,
            theta0,
            args=(xtr, phi, phi_bar_star, max_path_length),
            method="L-BFGS-B",
            jac=True,
            bounds=reward_parameter_bounds,
            options=minimize_options,
            **(minimize_kwargs),
        )
        x_star = res_lbfgs.x

        print(res_lbfgs)
        print(x_star)

        # e.dot_output()

    print("Done")


if __name__ == "__main__":
    main()
