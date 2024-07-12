"""Script for uploading the collected demos."""
import logging
import os
import warnings
import zipfile
from typing import Optional

import numpy as np
import tempfile
from pathlib import Path
from copy import deepcopy

import wget
from tqdm import tqdm

import bigym
from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from bigym.const import CACHE_PATH, RELEASES_PATH
from demonstrations.utils import Metadata, ObservationMode
from demonstrations.demo import Demo, LightweightDemo
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

    _DEMOS = "demonstrations"
    _VERSION = bigym.__version__
    _LOCK = ".lock"

    def __init__(self, cache_root: Optional[Path] = None):
        """Init."""
        self._cache_root = cache_root or CACHE_PATH
        self._cache_path: Path = self._cache_root / self._VERSION / self._DEMOS
        self._cache_path.mkdir(parents=True, exist_ok=True)

    def add_files(self, file_paths: list[Path]):
        """Add the safetensors to the store.

        :param file_paths: List of paths to the safetensors.
        """
        for file_path in file_paths:
            self._add_file(file_path)

    def _add_file(self, demo_path: Path):
        new_demo_path = self._create_path(Metadata.from_safetensors(demo_path))
        new_demo_path.parent.mkdir(parents=True, exist_ok=True)
        new_demo_path.write_bytes(demo_path.read_bytes())

    def add_demos(self, demos: list[Demo]):
        """Add demos to the store.

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
                    self._add_demo(demo, temp_dir)
                    pbar.update()

    def add_demo(self, demo: Demo):
        """Add a demo to the store.

        :param demo: The demo to upload.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            self._add_demo(demo, Path(temp_dir))

    def _add_demo(self, demo: Demo, local_dir: Path):
        if isinstance(local_dir, str):
            local_dir = Path(local_dir)
        if demo.metadata.observation_mode != ObservationMode.Lightweight:
            if not self.lightweight_demo_exists(demo.metadata):
                lightweight_demo = LightweightDemo.from_demo(demo)
                self._add_demo(lightweight_demo, local_dir)
        if self.demo_exists(demo.metadata):
            return
        file_path = demo.save(local_dir / demo.metadata.filename)
        self._add_file(file_path)

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
        demos = self._get_demos(remote_dir, amount)
        # If requested demos do not exist, try to get the lightweight demos
        if not demos and metadata.observation_mode != ObservationMode.Lightweight:
            light_metadata = deepcopy(metadata)
            light_metadata.observation_mode = ObservationMode.Lightweight
            remote_dir = self._create_path(light_metadata).parent
            demos = self._get_demos(remote_dir, amount)
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

    def _get_demos(self, demos_directory: Path, amount: int) -> list[Demo]:
        self._cache_demos()
        if not demos_directory.exists():
            return []
        files = [file for file in demos_directory.iterdir()]
        if amount > len(files):
            raise TooManyDemosRequestedError(amount, len(files))
        elif amount > 0:
            np.random.shuffle(files)
            files = files[:amount]
        return [Demo.from_safetensors(file) for file in files]

    def _cache_demos(self):
        if self.cached:
            logging.info(f"Using cached demonstrations from: {self._cache_path}")
            return
        url = f"{RELEASES_PATH}/v{self._VERSION}/{self._DEMOS}.zip"
        logging.info(
            f"Demonstrations for v{self._VERSION} not found. "
            f"Downloading from: {url}"
        )
        local_filename = wget.download(url)
        if not zipfile.is_zipfile(local_filename):
            raise RuntimeError(f"Invalid demonstrations file: {local_filename}")
        with zipfile.ZipFile(local_filename, "r") as zip_ref:
            zip_ref.extractall(self._cache_path.parent)
        os.remove(local_filename)
        self.cached = True

    @property
    def cached(self):
        """Return True if demos are cached locally, else False."""
        lock_file = self._cache_path / self._LOCK
        return lock_file.exists()

    @cached.setter
    def cached(self, value: bool):
        """Set local cache status."""
        lock_file = self._cache_path / self._LOCK
        if value:
            lock_file.touch(exist_ok=True)
        elif lock_file.exists():
            os.remove(lock_file)

    def list_demo_paths(self, metadata: Metadata) -> list[Path]:
        """List the demos matching the metadata."""
        demo_dir = self._create_path(metadata).parent
        if not demo_dir.exists():
            warnings.warn(f"No demos found for {metadata}.")
            return []
        return [p for p in demo_dir.iterdir()]

    def _create_path(self, metadata: Metadata) -> Path:
        path = self._cache_path
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
