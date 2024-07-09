"""Script for uploading the collected demos."""
import warnings

import numpy as np
import tempfile
from cloudpathlib import CloudPath
from pathlib import Path
from copy import deepcopy

from tqdm import tqdm

from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from demonstrations.utils import Metadata, ObservationMode
from demonstrations.demo import Demo, LightweightDemo
from demonstrations.const import SAFETENSORS_SUFFIX
from demonstrations.demo_converter import DemoConverter


class DemoNotFoundError(Exception):
    """Exception raised when a demo is not found."""

    def __init__(self, metadata: Metadata):
        """Init.

        Args:
            metadata(Metadata): The metadata for the demo.
        """
        self.metadata = metadata
        super().__init__(f"Demo not found for {metadata}.")


class TooManyDemosRequestedError(Exception):
    """Exception raised when too many demos are requested."""

    def __init__(self, requested: int, found: int):
        """Init.

        Args:
            requested(int): The number of demos requested.
            found(int): The number of demos found.
        """
        self.requested = requested
        self.found = found
        super().__init__(
            f"Requested {requested} demos, but only {found} found. "
            "Try requesting a smaller number."
        )


class DemoStore:
    """Class to help storing and retrieving demos from a database."""

    def __init__(self, path: CloudPath):
        """Init.

        :param path: The path to the store.
        """
        self._path = path

    @classmethod
    def google_cloud(cls, bucket_path: str = "rll_data/bigym"):
        """For accessing a google cloud bucket.

        :param bucket_path: Path to the bucket.
        """
        return cls(CloudPath(f"gs://{bucket_path}"))

    def upload_safetensors(self, file_paths: list[Path]):
        """Upload the safetensors to the store.

        :param file_paths: List of paths to the safetensors.
        """
        for file_path in file_paths:
            self._upload_safetensor(file_path)

    def _upload_safetensor(self, file_path: Path):
        """Upload the safetensor to the store.

        :param file_path: Path to the safetensor.
        """
        new_file = self._create_path(Metadata.from_safetensors(file_path))
        new_file.upload_from(file_path)

    def upload_demos(self, demos: list[Demo]):
        """Upload demos to the store.

        :param demos: List of demos to upload.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            with tqdm(
                total=len(demos),
                desc="Uploading Demos",
                unit="demo",
                leave=False,
                position=0,
            ) as pbar:
                for demo in demos:
                    self._upload_demo(demo, temp_dir)
                    pbar.update()

    def upload_demo(self, demo: Demo):
        """Upload a demo to the store.

        :param demo: The demo to upload.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            self._upload_demo(demo, Path(temp_dir))

    def _upload_demo(self, demo: Demo, local_dir: Path):
        """Upload the demo to the store.

        :param demo: The demo to upload.
        :param local_dir: The local directory to store the demo before uploading.
        """
        if isinstance(local_dir, str):
            local_dir = Path(local_dir)
        if demo.metadata.observation_mode != ObservationMode.Lightweight:
            if not self.lightweight_demo_exists(demo.metadata):
                lightweight_demo = LightweightDemo.from_demo(demo)
                self._upload_demo(lightweight_demo, local_dir)
        if self.demo_exists(demo.metadata):
            return
        new_file = self._create_path(demo.metadata)
        file_path = demo.save(local_dir / demo.metadata.filename)
        new_file.upload_from(file_path)

    def get_demos(
        self,
        metadata: Metadata,
        amount: int = -1,
        frequency: int = CONTROL_FREQUENCY_MAX,
        only_successful: bool = False,
    ) -> list[Demo]:
        """Download the demos matching the metadata.

        :param metadata: The metadata to match the demos.
        :param amount: The amount of demos to get.
            If < 0, all demonstrations are returned.
        :param frequency: Demo control frequency.
        :param only_successful: Return only successful demos.
        :param always_decimate: Decimate demo even if frequency is same.

        :return: The demos matching the metadata.
        """
        demos = []
        if amount == 0:
            return demos
        remote_dir = self._create_path(metadata).parent
        demos = self._download_demos(remote_dir, amount)
        # If requested demos do not exist, try to get the lightweight demos
        if not demos and metadata.observation_mode != ObservationMode.Lightweight:
            light_metadata = deepcopy(metadata)
            light_metadata.observation_mode = ObservationMode.Lightweight
            remote_dir = self._create_path(light_metadata).parent
            demos = self._download_demos(remote_dir, amount)
        # Raising exception if there are no demos at this stage
        if not demos:
            raise DemoNotFoundError(metadata)

        # Process downloaded demos
        processed_demos = []
        with tqdm(
            total=len(demos),
            desc="Processing Demos",
            unit="demo",
            leave=True,
            position=0,
        ) as pbar:
            robot = metadata.get_robot()
            env = metadata.get_env(frequency)
            for demo in demos:
                demo = DemoConverter.decimate(
                    demo,
                    frequency,
                    CONTROL_FREQUENCY_MAX,
                    robot=robot,
                )
                if (
                    metadata.observation_mode != ObservationMode.Lightweight
                    or only_successful
                ):
                    demo = DemoConverter.create_demo_in_new_env(demo, env)
                if not only_successful:
                    processed_demos.append(demo)
                elif np.any([bool(timestep.reward) for timestep in demo.timesteps]):
                    processed_demos.append(demo)
                pbar.update()
        return processed_demos

    @staticmethod
    def _download_demos(remote_dir: CloudPath, amount: int) -> list[Demo]:
        demos = []
        if not remote_dir.exists():
            return demos
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_dir.download_to(temp_dir)
            local_dir = Path(temp_dir)
            files = [file for file in remote_dir.iterdir()]

            # If the amount is -1, return all demos
            if amount > len(files):
                raise TooManyDemosRequestedError(amount, len(files))
            elif amount > 0:
                # If the amount is > 0 and < len(files), shuffle the files
                np.random.shuffle(files)
                files = files[:amount]

            # Download the demos
            for file in files:
                if file.suffix == SAFETENSORS_SUFFIX:
                    demos.append(Demo.from_safetensors(local_dir / file))
        return demos

    def list_demo_paths(self, metadata: Metadata) -> list[CloudPath]:
        """List the demos matching the metadata.

        :param metadata: The metadata to match the demos.

        :return: The paths to the demos matching the metadata.
        """
        remote_dir = self._create_path(metadata).parent
        if not remote_dir.exists():
            warnings.warn(f"No demos found for {metadata}.")
            return []
        return [p for p in remote_dir.iterdir()]

    def _create_path(self, metadata: Metadata) -> CloudPath:
        """Create a path for the demo.

        :param metadata: The metadata for the demo.

        :return: The path to the demo.
        """
        path = self._path
        if metadata.env_cls.DEFAULT_ROBOT != metadata.robot_cls:
            path /= metadata.environment_data.robot_name
        path = (
            path
            / metadata.env_name
            / metadata.environment_data.action_mode_description
            / metadata.observation_mode.value
        )
        if metadata.observation_mode == ObservationMode.Pixel:
            path /= metadata.environment_data.camera_description
        return path / metadata.filename

    def lightweight_demo_exists(self, metadata: Metadata) -> bool:
        """Check if a lightweight demo exists.

        :param metadata: The metadata for the demo.

        :return: True if the lightweight demo exists, False otherwise.
        """
        lightweight_metadata = deepcopy(metadata)
        lightweight_metadata.observation_mode = ObservationMode.Lightweight
        return self.demo_exists(lightweight_metadata)

    def demo_exists(self, metadata: Metadata) -> bool:
        """Check if a demo exists.

        :param metadata: The metadata for the demo.

        :return: True if the demo exists, False otherwise.
        """
        return self._create_path(metadata).exists()
