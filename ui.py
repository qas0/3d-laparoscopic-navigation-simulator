from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from astar import find_path
from geometry import movement_is_clear, shaft_clearance, tip_position
from scenarios import SCENARIOS
from sensors import measure_position


class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Laparoscopic Navigation Simulator")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 650)

        self.port = np.zeros(3)
        self.tool_radius = 1.5
        self.required_clearance = 2.0
        self.lower_limits = np.array([np.deg2rad(-45), np.deg2rad(-35), 10.0])
        self.upper_limits = np.array([np.deg2rad(45), np.deg2rad(35), 110.0])
        self.speed_limits = np.array([np.deg2rad(20), np.deg2rad(20), 15.0])
        self.target_tolerance = 2.0
        self.random_generator = np.random.default_rng()
        self.path = None
        self.path_actor = None
        self.path_index = 0
        self.following = False
        self.planning_future = None
        self.planning_cancel = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pressed_keys = set()
        self.key_directions = {
            Qt.Key.Key_A: (0, -1), Qt.Key.Key_D: (0, 1),
            Qt.Key.Key_Q: (1, -1), Qt.Key.Key_E: (1, 1),
            Qt.Key.Key_S: (2, -1), Qt.Key.Key_W: (2, 1),
        }

        self.setStyleSheet("""
            QWidget { background: #f6f8fa; color: #243447; font: 10pt 'Segoe UI'; }
            QLabel, QCheckBox { background: transparent; }
            QFrame#controls { background: white; border: 1px solid #d9e1e8; border-radius: 8px; }
            QGroupBox { background: white; font-weight: bold; border: 1px solid #d9e1e8;
                        border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background: white; border: 1px solid #bdcbd6;
                          border-radius: 5px; padding: 8px; }
            QPushButton:hover { background: #e8f0f6; }
            QPushButton:checked { background: #dbeaf5; border-color: #4688b6; }
            QPushButton:disabled { color: #8997a3; background: #f0f3f5; }
            QComboBox, QDoubleSpinBox, QSpinBox { background: white;
                        border: 1px solid #bdcbd6; border-radius: 5px; padding: 5px; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #e8eef3; padding: 8px 16px; }
            QTabBar::tab:selected { background: white; color: #286e9f; }
            QSlider::groove:horizontal { height: 5px; background: #d4dfe7; border-radius: 2px; }
            QSlider::handle:horizontal { background: #286e9f; width: 15px;
                                        margin: -5px 0; border-radius: 7px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 14)
        title = QLabel("3D Laparoscopic Navigation Simulator")
        title.setStyleSheet("font-size: 19pt; font-weight: 600;")
        layout.addWidget(title)
        layout.addWidget(QLabel("Static anatomy · Manual + A* navigation · Distances in millimetres"))
        body = QHBoxLayout()
        layout.addLayout(body, 1)

        controls = QFrame()
        controls.setObjectName("controls")
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setWidget(controls)
        self.control_tabs = QTabWidget()
        self.control_tabs.setFixedWidth(320)
        self.control_tabs.addTab(control_scroll, "Navigation")
        body.addWidget(self.control_tabs)

        scenario_title = QLabel("Scenario")
        scenario_title.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(scenario_title)
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(list(SCENARIOS))
        control_layout.addWidget(self.scenario_combo)
        self.scenario_description = QLabel()
        self.scenario_description.setWordWrap(True)
        control_layout.addWidget(self.scenario_description)

        instrument_group = QGroupBox("Instrument commands")
        instrument_group.setMinimumHeight(175)
        instrument_layout = QVBoxLayout(instrument_group)
        self.sliders = []
        self.command_labels = []
        for index, (name, minimum, maximum, value, unit) in enumerate([
            ("Yaw", -45, 45, 0, "°"),
            ("Pitch", -35, 35, 0, "°"),
            ("Insertion", 10, 110, 35, "mm"),
        ]):
            label = QLabel(f"{name}: {value:.1f} {unit}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(minimum * 10, maximum * 10)
            slider.setValue(value * 10)
            slider.setToolTip(f"Command {name.lower()}; actual position appears below.")
            slider.valueChanged.connect(lambda _, axis=index: self.set_command(axis))
            instrument_layout.addWidget(label)
            instrument_layout.addWidget(slider)
            self.command_labels.append(label)
            self.sliders.append(slider)
        control_layout.addWidget(instrument_group)

        planning_group = QGroupBox("A* navigation")
        planning_layout = QVBoxLayout(planning_group)
        self.plan_button = QPushButton("Plan route")
        self.plan_button.clicked.connect(self.plan_path)
        self.follow_button = QPushButton("Follow route")
        self.follow_button.setEnabled(False)
        self.follow_button.clicked.connect(self.follow_path)
        self.cancel_button = QPushButton("Cancel route")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.stop_path)
        route_buttons = QHBoxLayout()
        route_buttons.addWidget(self.plan_button)
        route_buttons.addWidget(self.follow_button)
        planning_layout.addLayout(route_buttons)
        planning_layout.addWidget(self.cancel_button)
        self.plan_label = QLabel("Grid: 5° yaw / pitch + 5 mm insertion\nTarget tolerance: 2 mm")
        self.plan_label.setWordWrap(True)
        planning_layout.addWidget(self.plan_label)
        control_layout.addWidget(planning_group)

        keyboard_help = QLabel("Hold W / S: insert / retract\n"
                               "Hold A / D: decrease / increase yaw\n"
                               "Hold Q / E: decrease / increase pitch\n"
                               "Manual commands cancel a route.\n\n"
                               "Mouse drag: orbit   ·   Wheel: zoom")
        keyboard_help.setWordWrap(True)
        control_layout.addWidget(keyboard_help)

        self.pause_button = QPushButton("Pause movement")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.set_paused)
        control_layout.addWidget(self.pause_button)
        self.reset_button = QPushButton("Reset instrument")
        self.reset_button.setToolTip("Places the instrument at a different random starting position.")
        self.reset_button.clicked.connect(self.reset_instrument)
        control_layout.addWidget(self.reset_button)
        view_button = QPushButton("Reset camera")
        view_button.clicked.connect(self.reset_camera)
        control_layout.addWidget(view_button)

        self.labels_checkbox = QCheckBox("Show geometry labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.toggled.connect(self.toggle_labels)
        control_layout.addWidget(self.labels_checkbox)
        control_layout.addStretch()
        scene_note = QLabel("Tool radius: 1.5 mm\nRequired shaft clearance: 2.0 mm\n"
                            "Protected structure: red sphere\n\n"
                            "The organ surface is a visual reference.")
        scene_note.setWordWrap(True)
        control_layout.addWidget(scene_note)

        sensor_controls = QFrame()
        sensor_controls.setObjectName("controls")
        sensor_layout = QVBoxLayout(sensor_controls)
        sensor_layout.setContentsMargins(16, 16, 16, 16)
        sensor_scroll = QScrollArea()
        sensor_scroll.setWidgetResizable(True)
        sensor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sensor_scroll.setWidget(sensor_controls)
        self.control_tabs.addTab(sensor_scroll, "Sensors")

        sensor_group = QGroupBox("Position measurements")
        sensor_settings = QFormLayout(sensor_group)
        self.tool_noise = QDoubleSpinBox()
        self.target_noise = QDoubleSpinBox()
        for setting, value in [(self.tool_noise, 1.0), (self.target_noise, 2.0)]:
            setting.setRange(0.0, 10.0)
            setting.setDecimals(1)
            setting.setSingleStep(0.1)
            setting.setSuffix(" mm")
            setting.setKeyboardTracking(False)
            setting.setValue(value)
            setting.setToolTip("Gaussian noise standard deviation for each coordinate.")
        sensor_settings.addRow("Tool noise σ", self.tool_noise)
        sensor_settings.addRow("Target noise σ", self.target_noise)
        self.sensor_seed = QSpinBox()
        self.sensor_seed.setRange(0, 999999)
        self.sensor_seed.setValue(42)
        self.sensor_seed.setKeyboardTracking(False)
        self.sensor_seed.setToolTip("Repeats the same sequence of sensor noise.")
        sensor_settings.addRow("Noise seed", self.sensor_seed)
        sensor_layout.addWidget(sensor_group)
        self.restart_sensor_button = QPushButton("Restart noise sequence")
        self.restart_sensor_button.clicked.connect(self.reset_measurements)
        sensor_layout.addWidget(self.restart_sensor_button)
        self.measurements_checkbox = QCheckBox("Show measurement markers")
        self.measurements_checkbox.setChecked(True)
        self.measurements_checkbox.toggled.connect(self.toggle_measurements)
        sensor_layout.addWidget(self.measurements_checkbox)
        sensor_note = QLabel("σ is the standard deviation for each coordinate.\n"
                            "Sampling rate: 10 Hz.\n\n"
                            "The seed repeats sensor noise. Instrument resets stay random.")
        sensor_note.setWordWrap(True)
        sensor_layout.addWidget(sensor_note)
        measurements_group = QGroupBox("Latest observations")
        measurements_layout = QVBoxLayout(measurements_group)
        self.tool_measurement_label = QLabel()
        self.target_measurement_label = QLabel()
        self.measurement_error_label = QLabel()
        for label in (self.tool_measurement_label, self.target_measurement_label,
                      self.measurement_error_label):
            label.setWordWrap(True)
            measurements_layout.addWidget(label)
        sensor_layout.addWidget(measurements_group)
        self.measurement_count_label = QLabel()
        sensor_layout.addWidget(self.measurement_count_label)
        sensor_layout.addStretch()
        sensor_baseline = QLabel("Markers show raw measurements.\n"
                                "Navigation currently uses true positions.")
        sensor_baseline.setWordWrap(True)
        sensor_layout.addWidget(sensor_baseline)
        for setting in (self.tool_noise, self.target_noise, self.sensor_seed):
            setting.valueChanged.connect(self.reset_measurements)

        self.viewport = QtInteractor(central, auto_update=False)
        body.addWidget(self.viewport.interactor, 1)
        self.viewport.set_background("#f2f6f9")

        self.position_label = QLabel()
        self.target_label = QLabel()
        self.clearance_label = QLabel()
        readouts = QHBoxLayout()
        for label in (self.position_label, self.target_label, self.clearance_label):
            label.setStyleSheet("padding: 10px; background: white; border: 1px solid #d9e1e8;")
            readouts.addWidget(label, 1)
        layout.addLayout(readouts)
        self.message_label = QLabel("Ready for manual movement.")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self.sensor_timer = QTimer(self)
        self.sensor_timer.timeout.connect(self.update_measurements)
        self.load_scenario(self.scenario_combo.currentIndex())
        self.scenario_combo.currentIndexChanged.connect(self.load_scenario)

        QApplication.instance().installEventFilter(self)
        self.last_tick = perf_counter()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_movement)
        self.timer.start(33)

    def load_scenario(self, index):
        self.scenario_name = self.scenario_combo.itemText(index)
        scenario = SCENARIOS[self.scenario_name]
        self.configuration = scenario["start"].copy()
        self.target = tip_position(self.port, *scenario["target_configuration"])
        self.structure_centre = scenario["structure_centre"].copy()
        self.structure_radius = scenario["structure_radius"]
        # places the target on the near surface of the reference ellipsoid
        self.organ_centre = self.target + np.array([20.0, 0.0, 0.0])
        self.stop_path()
        self.pressed_keys.clear()
        self.pause_button.setChecked(False)
        self.scenario_description.setText(scenario["description"])
        self.viewport.clear()
        self.build_scene()
        self.update_instrument()
        self.reset_measurements()
        self.message_label.setText(f"{self.scenario_name} loaded. Plan a route or use manual controls.")

    def build_scene(self):
        organ = pv.Sphere(radius=1, theta_resolution=64, phi_resolution=48)
        organ.points = organ.points * np.array([20, 26, 22]) + self.organ_centre
        self.viewport.add_mesh(organ, color="#d5968c", opacity=0.55, smooth_shading=True)
        self.viewport.add_mesh(pv.Sphere(radius=self.structure_radius, center=self.structure_centre),
                               color="#bc4046", smooth_shading=True)
        margin = pv.Sphere(radius=self.structure_radius + self.required_clearance,
                           center=self.structure_centre)
        self.viewport.add_mesh(margin, color="#ce7a7e", style="wireframe", opacity=0.22)
        self.viewport.add_mesh(pv.Sphere(radius=2.2, center=self.target),
                               color="#199a78", smooth_shading=True)
        self.viewport.add_mesh(pv.Sphere(radius=3, center=self.port), color="#326f9e")

        tip = tip_position(self.port, *self.configuration)
        self.shaft_actor = self.viewport.add_mesh(
            pv.Line(self.port, tip).tube(radius=self.tool_radius, n_sides=24, capping=True),
            color="#526b80", smooth_shading=True)
        self.tip_actor = self.viewport.add_mesh(pv.Sphere(radius=self.tool_radius),
                                                color="#d59420", smooth_shading=True)
        self.tool_measurement_actor = self.viewport.add_mesh(
            pv.Sphere(radius=2.0, theta_resolution=16, phi_resolution=12),
            color="#287dc0", style="wireframe", line_width=2)
        self.target_measurement_actor = self.viewport.add_mesh(
            pv.Sphere(radius=2.4, theta_resolution=16, phi_resolution=12),
            color="#b050a2", style="wireframe", line_width=2)
        for actor in (self.tool_measurement_actor, self.target_measurement_actor):
            actor.SetVisibility(self.measurements_checkbox.isChecked())
        self.label_actor = self.viewport.add_point_labels(
            np.array([self.port, self.target, self.structure_centre,
                      self.organ_centre + np.array([0.0, 0.0, 22.0])]),
            ["Fixed port", "Surface target", "Protected structure", "Organ surface"],
            font_size=13, text_color="#243447", point_size=0, shape_opacity=0.8,
            always_visible=True)
        self.label_actor.SetVisibility(self.labels_checkbox.isChecked())
        self.viewport.add_axes(color="#243447")
        self.viewport.add_legend([
            ["Instrument", "#526b80", "line"], ["Tool tip", "#d59420", "circle"],
            ["Target", "#199a78", "circle"], ["Protected structure", "#bc4046", "circle"],
            ["Planned tip trajectory", "#7757b5", "line"],
            ["Measured tool tip", "#287dc0", "circle"],
            ["Measured target", "#b050a2", "circle"],
        ], bcolor="white", border=False, size=(0.31, 0.22), loc="upper left", font_family="arial")
        self.reset_camera()

    def reset_camera(self):
        self.viewport.camera_position = [(170, -190, 145), (65, 0, 0), (0, 0, 1)]
        self.viewport.reset_camera()
        self.viewport.render()

    def plan_path(self):
        self.stop_path()
        self.pressed_keys.clear()
        self.command = self.configuration.copy()
        self.sync_controls()
        self.planning_cancel = Event()
        # searches with copied scene values; only the UI thread updates widgets
        self.planning_future = self.executor.submit(
            find_path, self.configuration.copy(), self.target.copy(), self.port.copy(),
            self.tool_radius, self.structure_centre.copy(), self.structure_radius,
            self.lower_limits.copy(), self.upper_limits.copy(), self.required_clearance,
            self.target_tolerance, cancel_event=self.planning_cancel)
        self.plan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.plan_label.setText("Searching the configuration grid…")
        self.message_label.setText("Planning a route. Cancel or use manual controls to interrupt.")

    def poll_planning(self):
        if self.planning_future is None or not self.planning_future.done():
            return
        future = self.planning_future
        self.planning_future = None
        self.planning_cancel = None
        self.plan_button.setEnabled(True)
        try:
            result = future.result()
        except Exception as error:
            self.cancel_button.setEnabled(False)
            self.plan_label.setText("Planning failed.")
            self.message_label.setText(f"Planning failed: {error}")
            return
        self.plan_label.setText(f"Search: {result['time']:.3f} s\n"
                               f"Expanded configurations: {result['expanded']}")
        if result["status"] != "found":
            messages = {
                "invalid_start": "The starting configuration violates limits or required clearance.",
                "invalid_target": "The target region is inaccessible within the limits or required clearance.",
                "no_grid_goal": "No configuration on this grid meets the target tolerance and clearance.",
                "no_path": "No route connects the start to a valid goal on this grid.",
                "budget_exceeded": "Search budget exhausted; a route may still exist.",
                "cancelled": "Planning cancelled.",
            }
            self.cancel_button.setEnabled(False)
            self.message_label.setText(messages[result["status"]])
            return
        self.path = result["path"]
        self.plan_label.setText(f"Planned tip travel: {result['cost']:.2f} mm\n"
                               f"Search: {result['time']:.3f} s · Expanded: {result['expanded']}")
        if len(self.path) == 1:
            self.cancel_button.setEnabled(False)
            self.message_label.setText("The tip is already within the target tolerance.")
            return
        points = [tip_position(self.port, *self.path[0])]
        for start, end in zip(self.path, self.path[1:]):
            change = end - start
            displacement = abs(change[2]) + max(start[2], end[2]) * sum(abs(change[:2]))
            intervals = max(1, int(np.ceil(displacement / 2.0)))
            for fraction in np.linspace(0, 1, intervals + 1)[1:]:
                points.append(tip_position(self.port, *(start + fraction * change)))
        # samples rotational arcs so the display follows the planned tip movement
        self.path_actor = self.viewport.add_mesh(
            pv.lines_from_points(np.array(points)), color="#7757b5", line_width=4)
        self.viewport.render()
        self.follow_button.setEnabled(True)
        self.message_label.setText("Route ready. Press Follow route to begin autonomous movement.")

    def follow_path(self):
        if self.path is None or len(self.path) < 2:
            return
        if not np.allclose(self.configuration, self.path[0], atol=1e-9, rtol=0):
            self.stop_path()
            self.message_label.setText("The instrument moved. Plan a new route from its current position.")
            return
        self.pressed_keys.clear()
        self.pause_button.setChecked(False)
        self.command = self.configuration.copy()
        self.path_index = 1
        self.following = True
        self.plan_button.setEnabled(False)
        self.follow_button.setEnabled(False)
        self.message_label.setText("Following the planned route.")

    def stop_path(self):
        if self.planning_cancel is not None:
            self.planning_cancel.set()
        if self.planning_future is not None:
            self.planning_future.cancel()
        self.planning_future = None
        self.planning_cancel = None
        self.following = False
        self.path = None
        self.path_index = 0
        self.command = self.configuration.copy()
        if self.path_actor is not None:
            self.viewport.remove_actor(self.path_actor)
            self.path_actor = None
        self.plan_button.setEnabled(True)
        self.follow_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.plan_label.setText("Grid: 5° yaw / pitch + 5 mm insertion\nTarget tolerance: 2 mm")
        self.sync_controls()
        self.message_label.setText("Route cancelled. Ready for manual movement.")

    def set_command(self, axis):
        value = self.sliders[axis].value() / 10
        if self.path is not None or self.planning_future is not None:
            self.stop_path()
        self.command[axis] = np.deg2rad(value) if axis < 2 else value
        self.sync_controls()

    def sync_controls(self):
        values = [np.rad2deg(self.command[0]), np.rad2deg(self.command[1]), self.command[2]]
        for index, (name, unit) in enumerate([("Yaw", "°"), ("Pitch", "°"), ("Insertion", "mm")]):
            self.sliders[index].blockSignals(True)
            self.sliders[index].setValue(round(values[index] * 10))
            self.sliders[index].blockSignals(False)
            self.command_labels[index].setText(f"{name}: {values[index]:.1f} {unit}")

    def set_paused(self, paused):
        self.pressed_keys.clear()
        self.pause_button.setText("Resume movement" if paused else "Pause movement")
        if self.following:
            self.message_label.setText("Route paused." if paused else "Resuming the planned route.")
        else:
            self.message_label.setText("Movement paused." if paused else "Ready for manual movement.")

    def reset_instrument(self):
        self.stop_path()
        self.pressed_keys.clear()
        self.pause_button.setChecked(False)
        current_tip = tip_position(self.port, *self.configuration)
        # samples a different clear position within the movement limits
        for _ in range(1000):
            candidate = self.random_generator.uniform(self.lower_limits, self.upper_limits)
            # matches the slider precision: 0.1 degrees + 0.1 mm
            candidate[:2] = np.deg2rad(np.round(np.rad2deg(candidate[:2]), 1))
            candidate[2] = round(candidate[2], 1)
            tip = tip_position(self.port, *candidate)
            if np.linalg.norm(tip - current_tip) < 10.0:
                continue
            if np.linalg.norm(tip - self.target) <= self.target_tolerance:
                continue
            if shaft_clearance(self.port, tip, self.tool_radius, self.structure_centre,
                               self.structure_radius) >= self.required_clearance:
                self.configuration = candidate
                break
        else:
            self.message_label.setText("No different clear starting position was found. Try resetting again.")
            return
        self.command = self.configuration.copy()
        self.sync_controls()
        self.scenario_description.setText("Random start. Target and protected structure stay fixed.")
        self.update_instrument()
        self.viewport.reset_camera()
        self.reset_measurements()
        self.message_label.setText("Instrument reset to a new random starting position. Plan a new route or use manual controls.")

    def reset_measurements(self):
        # restarts sensor noise separately from random instrument positions
        self.sensor_generator = np.random.default_rng(self.sensor_seed.value())
        self.measurement_count = 0
        self.sensor_timer.start(100)
        self.update_measurements()

    def update_measurements(self):
        tool_tip = tip_position(self.port, *self.configuration)
        self.tool_measurement = measure_position(
            tool_tip, self.tool_noise.value(), self.sensor_generator)
        self.target_measurement = measure_position(
            self.target, self.target_noise.value(), self.sensor_generator)
        self.measurement_count += 1
        self.tool_measurement_actor.SetPosition(*self.tool_measurement)
        self.target_measurement_actor.SetPosition(*self.target_measurement)
        self.tool_measurement_label.setText(
            "Measured tool tip (mm)\n" + " / ".join(
                f"{axis} {value:.2f}" for axis, value in zip("XYZ", self.tool_measurement)))
        self.target_measurement_label.setText(
            "Measured target (mm)\n" + " / ".join(
                f"{axis} {value:.2f}" for axis, value in zip("XYZ", self.target_measurement)))
        tool_error = np.linalg.norm(self.tool_measurement - tool_tip)
        target_error = np.linalg.norm(self.target_measurement - self.target)
        self.measurement_error_label.setText(
            f"Noise error at sample (mm)\nTool {tool_error:.2f} · Target {target_error:.2f}")
        self.measurement_count_label.setText(f"Measurement samples: {self.measurement_count}")
        self.viewport.render()

    def toggle_measurements(self, visible):
        self.tool_measurement_actor.SetVisibility(visible)
        self.target_measurement_actor.SetVisibility(visible)
        self.viewport.render()

    def toggle_labels(self, visible):
        self.label_actor.SetVisibility(visible)
        self.viewport.render()

    def advance_movement(self):
        now = perf_counter()
        # limits delayed frames to prevent sudden movement jumps
        timestep = min(now - self.last_tick, 0.05)
        self.last_tick = now
        self.poll_planning()
        if self.pause_button.isChecked() or self.planning_future is not None:
            return

        if self.pressed_keys:
            velocity = np.zeros(3)
            for key in self.pressed_keys:
                axis, direction = self.key_directions[key]
                velocity[axis] += direction * self.speed_limits[axis]
            self.command = np.clip(self.configuration + velocity * timestep,
                                   self.lower_limits, self.upper_limits)
            self.sync_controls()

        if self.following:
            remaining = self.path[self.path_index] - self.configuration
            duration = float(np.max(abs(remaining) / self.speed_limits))
            # preserves the same linear configuration segment checked by the planner
            fraction = min(1.0, timestep / max(duration, 1e-12))
            change = remaining * fraction
        else:
            change = np.clip(self.command - self.configuration,
                             -self.speed_limits * timestep, self.speed_limits * timestep)
        # advances reached waypoints even when rounding leaves zero movement
        if not np.any(change) and not self.following:
            return
        proposed = self.configuration + change
        if movement_is_clear(self.port, self.configuration, proposed, self.tool_radius,
                             self.structure_centre, self.structure_radius,
                             self.required_clearance):
            self.configuration = proposed
            self.update_instrument()
            if self.following:
                self.command = self.configuration.copy()
                self.sync_controls()
                if fraction == 1.0 or np.array_equal(self.configuration, self.path[self.path_index]):
                    self.path_index += 1
                if self.path_index == len(self.path):
                    self.following = False
                    self.plan_button.setEnabled(True)
                    distance = np.linalg.norm(tip_position(self.port, *self.configuration) - self.target)
                    if distance <= self.target_tolerance + 1e-9:
                        self.message_label.setText(f"Target reached · tip–target distance {distance:.2f} mm.")
                    else:
                        self.message_label.setText("Route finished outside the target tolerance. Plan a new route.")
                else:
                    self.message_label.setText(
                        f"Following route · waypoint {self.path_index} of {len(self.path) - 1}.")
            else:
                self.message_label.setText("Manual movement · shaft clearance maintained.")
        else:
            if self.following:
                self.stop_path()
                self.message_label.setText("Route stopped: the required shaft clearance cannot be assured.")
                return
            self.command = self.configuration.copy()
            self.sync_controls()
            self.message_label.setText("Movement blocked: the required shaft clearance cannot be assured.")

    def update_instrument(self):
        tip = tip_position(self.port, *self.configuration)
        self.shaft_actor.mapper.dataset = pv.Line(self.port, tip).tube(
            radius=self.tool_radius, n_sides=24, capping=True)
        self.tip_actor.SetPosition(*tip)
        target_distance = np.linalg.norm(tip - self.target)
        clearance = shaft_clearance(self.port, tip, self.tool_radius,
                                    self.structure_centre, self.structure_radius)
        yaw, pitch = np.rad2deg(self.configuration[:2])
        self.position_label.setText(f"Actual tool configuration\nYaw {yaw:.1f}° · Pitch {pitch:.1f}° · Depth {self.configuration[2]:.1f} mm")
        self.target_label.setText(f"Tip–target distance\n{target_distance:.2f} mm")
        self.clearance_label.setText(f"Whole-shaft clearance\n{clearance:.2f} mm   (required: {self.required_clearance:.1f} mm)")
        self.viewport.render()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ApplicationDeactivate or (
                watched is self and event.type() == QEvent.Type.WindowDeactivate):
            self.pressed_keys.clear()
        if isinstance(watched, QWidget) and watched.window() is self:
            if event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress,
                                QEvent.Type.KeyRelease) and event.key() in self.key_directions:
                if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
                    return super().eventFilter(watched, event)
                if event.type() == QEvent.Type.ShortcutOverride:
                    event.accept()
                elif not event.isAutoRepeat():
                    if event.type() == QEvent.Type.KeyPress and not self.pause_button.isChecked():
                        if self.path is not None or self.planning_future is not None:
                            self.stop_path()
                        self.pressed_keys.add(event.key())
                    else:
                        self.pressed_keys.discard(event.key())
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        self.timer.stop()
        self.sensor_timer.stop()
        if self.planning_cancel is not None:
            self.planning_cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
        QApplication.instance().removeEventFilter(self)
        self.viewport.close()
        super().closeEvent(event)
