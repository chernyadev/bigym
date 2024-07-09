"""Convert and upload old demos to the cloud."""
from pathlib import Path
from safetensors import safe_open

from bigym.action_modes import JointPositionActionMode
from bigym.bigym_env import BiGymEnv
from bigym.envs.move_plates import MovePlate
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from demonstrations.demo import Demo, LightweightDemo, DemoStep
from demonstrations.demo_store import DemoStore
from demonstrations.utils import ObservationMode, Metadata
from demonstrations.const import ACTION_KEY


def create_new_lightweight_demo(env: BiGymEnv, path: Path) -> LightweightDemo:
    """Create a new lightweight demo from old metadata and timesteps."""
    timesteps = Demo.load_timesteps_from_safetensors(path)
    timesteps = [DemoStep(*step, step[-1][ACTION_KEY]) for step in timesteps]
    with safe_open(path, framework="np", device="cpu") as f:
        old_metadata = f.metadata() or {}

    metadata = Metadata.from_env(env)
    metadata.observation_mode = ObservationMode.Lightweight
    metadata.seed = int(old_metadata["seed"])
    return LightweightDemo(metadata=metadata, timesteps=timesteps)


# Load the demos from the tmp folder and upload them to the cloud
path = Path(__file__).parent.parent / "tmp/plate"
env = MovePlate(
    action_mode=JointPositionActionMode(absolute=True),
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig(
                name="head",
                resolution=(16, 16),
            )
        ],
    ),
)

demo_store = DemoStore.google_cloud()
demos = demo_store.get_demos(metadata=Metadata.from_env(env), amount=1)

# Convert demos
# lightweight_demos = []
# for path in path.rglob(f"*{SAFETENSORS_SUFFIX}"):
#     lightweight_demos.append(
#         create_new_lightweight_demo(env, path)
#     )

# Save lightweight demos
# path = Path(__file__).parent.parent / "tmp/lightweight"
# for demo in lightweight_demos:
#     demo.save(path / demo.metadata.filename)

# Upload lightweight demos
# demo_store = DemoStore.google_cloud()
# demo_store.upload_safetensors(list(path.rglob(f"*{SAFETENSORS_SUFFIX}")))
