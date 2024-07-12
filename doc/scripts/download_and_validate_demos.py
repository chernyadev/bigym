import csv
from multiprocessing import Pool
from pathlib import Path
from typing import Type


from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MIN, BiGymEnv
from bigym.envs.reach_target import ReachTarget, ReachTargetDual, ReachTargetSingle
from demonstrations.demo_store import DemoStore, DemoNotFoundError
from demonstrations.demo_player import DemoPlayer
from demonstrations.utils import Metadata
from tools.shared.utils import ENVIRONMENTS

PROCESSES = 12
ACTION_MODES = [
    JointPositionActionMode(
        absolute=True,
        floating_base=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    ),
    JointPositionActionMode(
        absolute=True,
        floating_base=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.RZ],
    ),
]
ENVIRONMENTS_TO_SKIP = {ReachTarget, ReachTargetDual, ReachTargetSingle}

output_directory = Path(__file__).resolve().parent / "validation"
demo_store = DemoStore.google_cloud()


def validate_all_demos(env_cls: Type[BiGymEnv]):
    for action_mode in ACTION_MODES:
        env = env_cls(action_mode=action_mode, control_frequency=CONTROL_FREQUENCY_MIN)
        try:
            print(f"{env.task_name}: trying to download demos...")
            demos = demo_store.get_demos(
                metadata=Metadata.from_env(env, is_lightweight=True)
            )
            print(f"{env.task_name}: downloaded {len(demos)} demos")
            for demo in demos:
                demo.save(
                    output_directory / demo.metadata.env_name / demo.metadata.filename
                )
        except DemoNotFoundError:
            continue

        with open(
            output_directory / f"{env.task_name}_validation.csv", "w", newline=""
        ) as csvfile:
            log_writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                log_writer.writerow(["Filename", "Valid"])
                csvfile.flush()
            for i, demo in enumerate(demos):
                demo_is_valid = DemoPlayer.validate_in_env(demo, env)
                log_writer.writerow([demo.metadata.filename, demo_is_valid])
                csvfile.flush()
                print(f"\n{env.task_name}: validated {i+1}/{len(demos)} demos")
        return


envs_to_validate = [
    cls for cls in ENVIRONMENTS.values() if cls not in ENVIRONMENTS_TO_SKIP
]
with Pool(processes=PROCESSES) as pool:
    pool.map(validate_all_demos, envs_to_validate)
