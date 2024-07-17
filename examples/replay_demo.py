"""Retrieve and replay demonstration from the dataset.

Notes:
    - On the first run, the latest demos are downloaded and saved locally.
    - Demos are re-recorded at `control_frequency` and cached locally on the first run.
"""

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

control_frequency = 50
env = ReachTarget(
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    control_frequency=50,
    render_mode="human",
)
metadata = Metadata.from_env(env)

# Get demonstrations from DemoStore
demo_store = DemoStore()
demos = demo_store.get_demos(metadata, amount=1, frequency=control_frequency)

# Replay first demonstration
player = DemoPlayer()
player.replay_in_env(demos[0], env, demo_frequency=control_frequency)
