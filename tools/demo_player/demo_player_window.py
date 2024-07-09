"""Demo Player GUI."""
import multiprocessing
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Type

import glfw
import numpy as np
from dearpygui import dearpygui as dpg

from bigym.action_modes import JointPositionActionMode, PelvisDof, DEFAULT_DOFS
from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX, CONTROL_FREQUENCY_MIN
from bigym.robots.robot import Robot
from demonstrations.demo import Demo, DemoStep, LightweightDemo
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo_store import DemoNotFoundError, DemoStore
from demonstrations.utils import Metadata, ObservationMode
from tools.demo_player.demo_player_rendering import DemoPlayerRenderer
from tools.shared.settings_shelf import SettingsShelf
from tools.shared.base_window import BaseWindow
from tools.demo_player.demo_converter_window import DemoConverterWindow
from tools.shared.utils import (
    ReplayMode,
    REPLAY_MODES,
    ENVIRONMENTS,
    get_demos_in_dir,
)
from demonstrations.demo_converter import DemoConverter


@dataclass
class DemosTableRow:
    """Demos table row."""

    demo_file: Path
    table_row: Union[int, str]
    checkbox: Union[int, str]
    termination_col: Union[int, str]
    reward_col: Union[int, str]
    stable_col: Union[int, str]


class DemosTable:
    """Demos Table."""

    def __init__(self):
        """Init."""
        self._rows: list[DemosTableRow] = []

        with dpg.theme() as table_theme:
            with dpg.theme_component(dpg.mvTable):
                dpg.add_theme_color(
                    dpg.mvThemeCol_HeaderActive,
                    (0, 0, 0, 0),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Header, (0, 0, 0, 0), category=dpg.mvThemeCat_Core
                )

        with dpg.child_window(height=400):
            with dpg.table(tag="SelectRows", header_row=True) as self._table:
                dpg.add_table_column(
                    width_fixed=True, width=20, init_width_or_weight=20
                )
                dpg.add_table_column(
                    label="No.", width_fixed=True, width=20, init_width_or_weight=20
                )
                dpg.add_table_column(
                    label="File", width_stretch=True, init_width_or_weight=0.0
                )
                dpg.add_table_column(
                    label="Termination",
                    width_fixed=True,
                    width=100,
                    init_width_or_weight=100,
                )
                dpg.add_table_column(
                    label="Reward",
                    width_fixed=True,
                    width=100,
                    init_width_or_weight=100,
                )
                dpg.add_table_column(
                    label="Stability",
                    width_fixed=True,
                    width=100,
                    init_width_or_weight=100,
                )
                dpg.bind_item_theme(self._table, table_theme)

    def populate(self, demo_files: list[Path]):
        """Fill table."""
        self.clean()
        for file in demo_files:
            self._add_row(file)

    def clean(self):
        """Clean table."""
        for row in self._rows:
            dpg.delete_item(row.table_row)
        self._rows.clear()

    def get_selected_demos(self) -> list[Path]:
        """Get selected demo files."""
        selected = []
        for row in self._rows:
            if dpg.get_value(row.checkbox):
                selected.append(row.demo_file)
        return selected

    def update_demo_data(
        self,
        demo_file: Path,
        termination: bool = False,
        reward: float = 0,
        stability: float = 0,
        select: bool = False,
    ):
        """Update demo information."""
        row = next((r for r in self._rows if r.demo_file == demo_file), None)
        if not row:
            return
        dpg.set_item_label(row.termination_col, str(termination))
        dpg.set_item_label(row.reward_col, f"{reward:.2f}")
        dpg.set_item_label(row.stable_col, f"{int(np.round(stability * 100))}%")
        dpg.set_value(row.checkbox, select)

    def _add_row(self, demo_file: Path):
        """Add table row."""
        with dpg.table_row(parent=self._table) as table_row:
            row_index = len(self._rows)
            checkbox = dpg.add_checkbox()
            dpg.add_selectable(
                label=str(row_index + 1),
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            dpg.add_selectable(
                label=demo_file.name,
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            termination_col = dpg.add_selectable(
                label="-",
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            reward_col = dpg.add_selectable(
                label="-",
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            stable_col = dpg.add_selectable(
                label="-",
                span_columns=True,
                callback=self._row_selected_callback,
                user_data=row_index,
            )
            self._rows.append(
                DemosTableRow(
                    demo_file,
                    table_row,
                    checkbox,
                    termination_col,
                    reward_col,
                    stable_col,
                )
            )

    def _row_selected_callback(self, sender, app_data, user_data: int):
        if user_data < 0 or user_data >= len(self._rows):
            return
        checkbox = self._rows[user_data].checkbox
        dpg.set_value(checkbox, not dpg.get_value(checkbox))


class DemoPlayerWindow(BaseWindow):
    """Demo Player GUI."""

    SETTINGS_FILE = "demo_player"
    CURRENT_DIR = "current_dir"
    ENV_ID = "env_id"

    VALIDATION_THREADS = 4

    def __init__(self):
        """Init."""
        self._settings = SettingsShelf(self.SETTINGS_FILE)
        self._validation_processes: list[multiprocessing.Process] = []
        self._is_playing = False
        self._current_dir_value = None
        super().__init__()

    def _setup_ui(self):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Select Directory", callback=self._select_directory_callback
            )
            dpg.add_button(label="Convert", callback=self._convert_demos_callback)
            dpg.add_button(label="Validate", callback=self._validate_demos_callback)
            dpg.add_button(label="Download", callback=self._download_demos_callback)
            dpg.add_button(label="Upload", callback=self._upload_demos_callback)
            dpg.add_button(
                label="Delete Selected", callback=self._try_delete_demo_callback
            )
        dpg.add_spacer()
        with dpg.group(horizontal=True):
            self._table = DemosTable()
        dpg.add_spacer()
        with dpg.group(horizontal=True, width=200):
            with dpg.group(horizontal=False):
                dpg.add_text("Environment")
                self.env_listbox = dpg.add_listbox(
                    num_items=10,
                    items=list(sorted(ENVIRONMENTS.keys())),
                )
            with dpg.group(horizontal=False):
                dpg.add_text("Replay Mode")
                self.replay_mode_listbox = dpg.add_listbox(
                    num_items=10,
                    items=list(REPLAY_MODES.keys()),
                )
            self._dof_checkboxes: dict[PelvisDof, int] = {}
            with dpg.group(horizontal=False):
                dpg.add_text("Pelvis DOFs")
                for dof in PelvisDof:
                    is_default = dof in DEFAULT_DOFS
                    self._dof_checkboxes[dof] = dpg.add_checkbox(
                        label=dof.value, default_value=is_default
                    )
            with dpg.group(horizontal=False):
                dpg.add_text("Control Frequency")
                self.frequency_entry = dpg.add_input_int(
                    default_value=CONTROL_FREQUENCY_MAX,
                    min_value=CONTROL_FREQUENCY_MIN,
                    max_value=CONTROL_FREQUENCY_MAX,
                    min_clamped=True,
                    max_clamped=True,
                    step=10,
                )
        dpg.add_spacer()
        with dpg.group(horizontal=True, width=200):
            dpg.add_button(
                label="Replay Selected",
                callback=self._replay_selected_callback,
                width=200,
            )
            dpg.add_text("Last replayed demo:")
            self.last_file_name = dpg.add_text("...")
        self._converter_window = DemoConverterWindow()
        self._update_files_list()

    def on_close(self):
        """See base."""
        self._stop_demo_replay()
        self._cancel_validation()

    def _stop_demo_replay(self):
        """Stop active demo replay."""
        if not self._is_playing:
            return
        self._is_playing = False

    def _start_demo_replay(self, demo_path: Path):
        """Replay a demo from a file.

        :param demo_path: The path to the demo file.
        """
        if not demo_path.is_file() or self._is_playing:
            return
        self._is_playing = True
        dpg.set_value(self.last_file_name, demo_path.name)
        self._run_demo(Demo.from_safetensors(demo_path))

    def _run_demo(self, demo: Demo):
        self._run_env(demo)
        # ToDo: Add visual observations replay

    def _run_env(self, demo: Demo):
        frequency = self._get_frequency()
        env = self._create_env(frequency, demo.metadata.robot_cls)
        demo = DemoConverter.decimate(demo, frequency, robot=env.robot)
        demo_renderer = DemoPlayerRenderer(env.mojo)
        env.mujoco_renderer = demo_renderer

        def render() -> bool:
            if (
                not self._is_playing
                or demo_renderer.viewer.window is None
                or glfw.window_should_close(demo_renderer.viewer.window)
            ):
                env.close()
                self._stop_demo_replay()
                return False
            else:
                env.render()
                return True

        while True:
            env.reset(seed=int(demo.seed))
            for timestep in demo.timesteps:
                try:
                    actual_timestep = DemoStep(
                        *env.step(timestep.executed_action), timestep.executed_action
                    )
                except ValueError as e:
                    warnings.warn(str(e))
                    env.close()
                    self._stop_demo_replay()
                    return
                demo_renderer.set_demo_data(
                    demo_info=timestep, actual_info=actual_timestep
                )
                if not render():
                    return

    def _select_directory_callback(self):
        self._select_directory(self._current_dir or "", self._on_directory_selected)

    def _on_directory_selected(self, selected_path: Path):
        self._current_dir = selected_path
        self._update_files_list()

    def _update_files_list(self):
        """Select the current directory."""
        current_dir = self._current_dir
        if current_dir is None:
            return
        demos = get_demos_in_dir(current_dir)
        self._table.populate(demos)

    def _replay_selected_callback(self):
        demos = self._table.get_selected_demos()
        if not demos:
            return
        demo = demos[-1]
        if not demo.is_file():
            return
        self._start_demo_replay(demo)

    def _try_delete_demo_callback(self):
        demos = self._table.get_selected_demos()
        if not demos:
            return
        message = "Delete selected demos?\n"
        message += "\n".join([str(demo.name) for demo in demos])
        self._show_popup(
            "Delete Demos",
            message,
            actions={
                "OK": self._demo_delete_callback,
                "Cancel": None,
            },
        )

    def _demo_delete_callback(self):
        demos = self._table.get_selected_demos()
        if not demos:
            return
        for demo in demos:
            os.remove(demo)
        self._update_files_list()

    def _convert_demos_callback(self):
        self._converter_window.show(self._current_dir)

    def _validate_demos_callback(self):
        results_queue = multiprocessing.Queue()
        popup = self._show_popup("Validate Demos", "Validating...")
        demo_files = get_demos_in_dir(self._current_dir)
        demo_batches: list[np.ndarray[Path]] = np.array_split(
            np.array(demo_files), self.VALIDATION_THREADS
        )
        progress_bar = dpg.add_progress_bar(width=-1, default_value=0.0, parent=popup)
        self._update_progress(progress_bar, 0, len(demo_files))
        dpg.add_button(
            label="OK", parent=popup, callback=lambda _: self._cancel_validation(popup)
        )
        popup_thread = threading.Thread(
            target=self._update_validation_popup,
            args=(popup, progress_bar, demo_files, results_queue),
        )
        for batch in demo_batches:
            demo = Demo.from_safetensors(batch.item())
            frequency = self._get_frequency()
            env = self._create_env(frequency, robot_cls=demo.metadata.robot_cls)
            p = multiprocessing.Process(
                target=self._validate_demos, args=(batch, frequency, env, results_queue)
            )
            self._validation_processes.append(p)
            p.start()
            if not popup_thread.is_alive():
                popup_thread.start()

    def _update_validation_popup(
        self,
        popup: Optional[Union[int, str]],
        progress_bar: Optional[Union[int, str]],
        demo_files: list[Path],
        results_queue: multiprocessing.Queue,
    ):
        results: list[tuple[Path, bool, float]] = []
        while dpg.does_item_exist(popup) and (
            any([p.is_alive() for p in self._validation_processes])
            or not results_queue.empty()
        ):
            while not results_queue.empty():
                demo_file, terminated, reward = results_queue.get()
                self._table.update_demo_data(
                    demo_file, terminated, reward, select=reward == 0
                )
                results.append((demo_file, terminated, reward))
            failed_demos = len([r for r in results if r[-1] == 0])
            self._update_progress(
                progress_bar, len(results), len(demo_files), f"{failed_demos} failed"
            )
            dpg.split_frame()

    def _cancel_validation(self, popup: Optional[Union[int, str]] = None):
        if popup:
            dpg.split_frame()
            dpg.delete_item(popup)
        for process in self._validation_processes:
            process.kill()
        self._validation_processes.clear()

    @staticmethod
    def _validate_demos(
        demo_files: np.ndarray[Path],
        env: BiGymEnv,
        results: multiprocessing.Queue,
    ):
        for i, demo_file in enumerate(demo_files):
            demo = Demo.from_safetensors(demo_file)
            DemoPlayer.validate_in_env(demo, env)
            results.put((demo_file, env.terminate, env.reward))
        env.close()

    @staticmethod
    def _update_progress(progress, current, total, note=""):
        dpg.set_value(progress, float(current) / total)
        overlay = f"{current}/{total}"
        if len(note):
            overlay += f" ({note})"
        dpg.configure_item(progress, overlay=overlay)

    def _download_demos_callback(self):
        popup = self._show_popup(
            "Download Demos", "Downloading...", loading_indicator=True
        )
        try:
            demos = self._download_demos(self._current_dir, self._create_env())
            dpg.delete_item(popup)
            dpg.split_frame()
            if demos:
                count = len(demos)
                self._update_files_list()
                self._show_popup(
                    "Download Demos",
                    f"Downloaded {count} demonstration{'' if count == 1 else 's'}.",
                    actions={"OK": None},
                )
        except DemoNotFoundError:
            dpg.delete_item(popup)
            dpg.split_frame()
            self._show_popup(
                "Download Demos", "No demonstrations found.", actions={"OK": None}
            )

    @staticmethod
    def _download_demos(
        directory: Optional[Path], env: BiGymEnv
    ) -> Optional[list[Demo]]:
        if not directory:
            return None
        demo_store = DemoStore.google_cloud()
        metadata = Metadata.from_env(env)
        metadata.observation_mode = ObservationMode.Lightweight
        demos = demo_store.get_demos(metadata, amount=-1, only_successful=False)
        for demo in demos:
            demo.save(directory / demo.metadata.filename)
        return demos

    def _upload_demos_callback(self):
        popup = self._show_popup(
            "Upload Demos", "Uploading demonstrations...", loading_indicator=True
        )
        demos = self._upload_demos(self._current_dir)
        dpg.delete_item(popup)
        dpg.split_frame()
        if demos:
            count = len(demos)
            self._show_popup(
                "Upload Demos",
                f"Uploaded {count} demonstration{'s' if count > 0 else ''}.",
                actions={"OK": None},
            )

    @staticmethod
    def _upload_demos(directory: Optional[Path]) -> Optional[list[Demo]]:
        if directory is None:
            return None
        demos = []
        for demo_file in get_demos_in_dir(directory):
            demo = Demo.from_safetensors(demo_file)
            demos.append(LightweightDemo.from_demo(demo))
        demo_store = DemoStore.google_cloud()
        demo_store.upload_demos(demos)
        return demos

    def _create_env(
        self,
        control_frequency=CONTROL_FREQUENCY_MAX,
        robot_cls: Optional[Type[Robot]] = None,
    ) -> BiGymEnv:
        replay_mode = REPLAY_MODES[dpg.get_value(self.replay_mode_listbox)]
        env_cls = ENVIRONMENTS[dpg.get_value(self.env_listbox)]
        pelvis_dofs = [
            dof
            for dof, checkbox in self._dof_checkboxes.items()
            if dpg.get_value(checkbox)
        ]
        action_mode = JointPositionActionMode(
            absolute=replay_mode == ReplayMode.Absolute, floating_dofs=pelvis_dofs
        )
        env = env_cls(
            action_mode=action_mode,
            render_mode="human",
            control_frequency=control_frequency,
            robot_cls=robot_cls,
        )
        return env

    def _get_frequency(self) -> int:
        return int(dpg.get_value(self.frequency_entry))

    @property
    def _current_dir(self) -> Optional[Path]:
        if self._current_dir_value is None:
            self._current_dir_value = self._settings.get(self.CURRENT_DIR, Path())
        if self._current_dir_value:
            if not isinstance(self._current_dir_value, Path):
                self._current_dir_value = Path(self._current_dir_value)
            if self._current_dir_value.exists():
                return self._current_dir_value
        return None

    @_current_dir.setter
    def _current_dir(self, value):
        self._current_dir_value = value
        self._settings.set(self.CURRENT_DIR, self._current_dir_value)
