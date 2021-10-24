REGISTRY = {}

from .vn_env import VNEnv
from .blockergame_env import BlockerGameEnv
REGISTRY["vn"] = VNEnv
REGISTRY["blocker"] = BlockerGameEnv


