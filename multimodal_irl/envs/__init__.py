from gym.envs.registration import register

from .puddle_world import *
from .element_world import *

register(
    id="PuddleWorld-v0",
    entry_point="puddle_world.envs:PuddleWorldEnv",
)

register(
    id="CanonicalPuddleWorld-v0",
    entry_point="puddle_world.envs:CanonicalPuddleWorldEnv",
)

register(
    id="SmallPuddleWorld-v0",
    entry_point="puddle_world.envs:SmallPuddleWorldEnv",
)

register(
    id="ElementWorld-v0",
    entry_point="element_world.envs:ElementWorldEnv",
)

