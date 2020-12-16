from gym.envs.registration import register

from .puddle_world import (
    PuddleWorldEnv,
    CanonicalPuddleWorldEnv,
    SmallPuddleWorldEnv,
    puddle_world_extras,
)

register(
    id="PuddleWorld-v0", entry_point="puddle_world.envs:PuddleWorldEnv",
)

register(
    id="CanonicalPuddleWorld-v0",
    entry_point="puddle_world.envs:CanonicalPuddleWorldEnv",
)

register(
    id="SmallPuddleWorld-v0", entry_point="puddle_world.envs:SmallPuddleWorldEnv",
)
