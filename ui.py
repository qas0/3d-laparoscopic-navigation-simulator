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
from kalman import KalmanFilter
from scenarios import SCENARIOS
from sensors import measure_position


class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Laparoscopic Navigation Simulator")
        self.resize(1360, 860)
        self.setMinimumSize(1000, 650)

        self.port = np.zeros(3)
        self.tool_radius = 1.5
        self.shaft_length = 135.0
        self.render_pending = False
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
            QWidget { background: #f3f6f9; color: #25384b; font: 10pt 'Segoe UI'; }
            QLabel, QCheckBox { background: transparent; }
            QCheckBox::indicator { width: 14px; height: 14px; background: white;
                                   border: 1px solid #9eb2c2; border-radius: 3px; }
            QCheckBox::indicator:checked { background: #286e9f; border-color: #286e9f; }
            QFrame#controls { background: white; border: 1px solid #d9e1e8; border-radius: 8px; }
            QFrame#scene { background: white; border: 1px solid #d9e1e8; border-radius: 8px; }
            QGroupBox { background: white; font-weight: bold; border: 1px solid #d9e1e8;
                        border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background: white; border: 1px solid #bdcbd6;
                          border-radius: 5px; padding: 8px; }
            QPushButton:hover { background: #e8f0f6; }
            QPushButton#primary { background: #286e9f; color: white; border-color: #286e9f; }
            QPushButton#primary:hover { background: #205d88; }
            QPushButton:checked { background: #dbeaf5; border-color: #4688b6; }
            QPushButton:disabled { color: #8997a3; background: #f0f3f5; }
            QComboBox, QDoubleSpinBox, QSpinBox { background: white;
                        border: 1px solid #bdcbd6; border-radius: 5px; padding: 5px; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #e8eef3; padding: 9px 13px; }
            QTabBar::tab:selected { background: white; color: #286e9f; }
            QSlider::groove:horizontal { height: 5px; background: #d4dfe7; border-radius: 2px; }
            QSlider::handle:horizontal { background: #286e9f; width: 15px;
                                        margin: -5px 0; border-radius: 7px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        title = QLabel("Laparoscopic navigation")
        title.setStyleSheet("font-size: 18pt; font-weight: 600;")
        layout.addWidget(title)
        layout.addWidget(QLabel("Fixed-port 3D simulator  ·  Manual control + A* planning  ·  Millimetres"))
        body = QHBoxLayout()
        layout.addLayout(body, 1)

        controls = QFrame()
        controls.setObjectName("controls")
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(12, 12, 12, 12)
        control_layout.setSpacing(10)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setWidget(controls)
        self.control_tabs = QTabWidget()
        self.control_tabs.setFixedWidth(330)
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
        self.plan_button.setObjectName("primary")
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
        self.plan_label.setMinimumHeight(52)
        planning_layout.addWidget(self.plan_label)
        control_layout.addWidget(planning_group)

        keyboard_help = QLabel("W / S  Insert / retract\n"
                               "A / D  Yaw       Q / E  Pitch\n"
                               "Hold a key to move. Manual control cancels a route.")
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
        control_layout.addStretch()
        scene_note = QLabel("Tool radius: 1.5 mm\nRequired shaft clearance: 2.0 mm\n"
                            "Protected structure: red sphere\n\n"
                            "The organ surface is a visual reference.")
        scene_note.setWordWrap(True)
        control_layout.addWidget(scene_note)

        sensor_controls = QFrame()
        sensor_controls.setObjectName("controls")
        sensor_layout = QVBoxLayout(sensor_controls)
        sensor_layout.setContentsMargins(12, 12, 12, 12)
        sensor_layout.setSpacing(10)
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
        self.measurements_checkbox = QCheckBox("Raw positions in scene")
        self.measurements_checkbox.setChecked(False)
        self.measurements_checkbox.setToolTip("Raw observations: blue tool + pink target. These markers are not physical objects.")
        self.measurements_checkbox.toggled.connect(self.toggle_measurements)
        sensor_layout.addWidget(self.measurements_checkbox)
        self.estimates_checkbox = QCheckBox("Filtered positions in scene")
        self.estimates_checkbox.setChecked(False)
        self.estimates_checkbox.setToolTip("Filtered crosses: navy tool + purple target. Their size does not represent uncertainty.")
        self.estimates_checkbox.toggled.connect(self.toggle_measurements)
        sensor_layout.addWidget(self.estimates_checkbox)
        sensor_note = QLabel("Sensors: 10 Hz  ·  Readouts: 2 Hz\n"
                            "σ applies to each coordinate. The seed repeats noise; instrument resets stay random.")
        sensor_note.setWordWrap(True)
        sensor_layout.addWidget(sensor_note)
        measurements_group = QGroupBox("Raw position readings · mm")
        measurements_layout = QVBoxLayout(measurements_group)
        self.tool_measurement_label = QLabel()
        self.target_measurement_label = QLabel()
        self.measurement_error_label = QLabel()
        for label in (self.tool_measurement_label, self.target_measurement_label,
                      self.measurement_error_label):
            label.setWordWrap(True)
            measurements_layout.addWidget(label)
        sensor_layout.addWidget(measurements_group)

        estimate_controls = QFrame()
        estimate_controls.setObjectName("controls")
        estimate_layout = QVBoxLayout(estimate_controls)
        estimate_layout.setContentsMargins(12, 12, 12, 12)
        estimate_layout.setSpacing(10)
        estimate_scroll = QScrollArea()
        estimate_scroll.setWidgetResizable(True)
        estimate_scroll.setFrameShape(QFrame.Shape.NoFrame)
        estimate_scroll.setWidget(estimate_controls)
        self.control_tabs.addTab(estimate_scroll, "Estimates")
        estimate_note = QLabel("Kalman filtering reduces measurement noise.\n"
                              "Readouts refresh together twice per second.")
        estimate_note.setWordWrap(True)
        estimate_layout.addWidget(estimate_note)
        filter_group = QGroupBox("Filtered position readings · mm")
        filter_layout = QVBoxLayout(filter_group)
        self.tool_estimate_label = QLabel()
        self.target_estimate_label = QLabel()
        self.estimate_error_label = QLabel()
        self.uncertainty_label = QLabel()
        self.rmse_label = QLabel()
        self.uncertainty_label.setToolTip("Model RMS position uncertainty: square root of the three position variances added together.")
        self.rmse_label.setToolTip("3D position errors over all samples since the last sensor restart, including the first reading.")
        for label in (self.tool_estimate_label, self.target_estimate_label,
                      self.estimate_error_label, self.uncertainty_label, self.rmse_label):
            label.setWordWrap(True)
            filter_layout.addWidget(label)
        estimate_layout.addWidget(filter_group)
        self.measurement_count_label = QLabel()
        sensor_layout.addWidget(self.measurement_count_label)
        sensor_layout.addStretch()
        estimate_layout.addStretch()
        marker_note = QLabel("Readings stay in the side panel by default.\n"
                             "Raw: blue tool / pink target\n"
                             "Filtered: navy tool / purple target")
        marker_note.setWordWrap(True)
        sensor_layout.addWidget(marker_note)
        sensor_baseline = QLabel("Constant-velocity filtering.\n"
                                "Motion σa: tool 20 / target 1 mm/s².\n"
                                "Errors use the known positions for evaluation.\n"
                                "Navigation currently uses true positions.")
        sensor_baseline.setWordWrap(True)
        estimate_layout.addWidget(sensor_baseline)
        for setting in (self.tool_noise, self.target_noise, self.sensor_seed):
            setting.valueChanged.connect(self.reset_measurements)

        scene_frame = QFrame()
        scene_frame.setObjectName("scene")
        scene_layout = QVBoxLayout(scene_frame)
        scene_layout.setContentsMargins(1, 1, 1, 1)
        scene_layout.setSpacing(0)
        scene_toolbar = QHBoxLayout()
        scene_toolbar.setContentsMargins(12, 8, 12, 8)
        scene_title = QLabel("3D WORKSPACE")
        scene_title.setStyleSheet("font-size: 9pt; font-weight: 600; color: #62788c;")
        scene_toolbar.addWidget(scene_title)
        scene_toolbar.addStretch()
        self.labels_checkbox = QCheckBox("Labels")
        self.labels_checkbox.setChecked(False)
        self.labels_checkbox.toggled.connect(self.toggle_labels)
        scene_toolbar.addWidget(self.labels_checkbox)
        self.view_combo = QComboBox()
        self.view_combo.setToolTip("Perspective for the task; Overview includes the full instrument movement range.")
        self.view_combo.addItems(["Perspective", "Top", "Front", "Overview"])
        self.view_combo.currentIndexChanged.connect(self.reset_camera)
        scene_toolbar.addWidget(self.view_combo)
        view_button = QPushButton("Fit view")
        view_button.clicked.connect(self.reset_camera)
        scene_toolbar.addWidget(view_button)
        scene_layout.addLayout(scene_toolbar)
        self.viewport = QtInteractor(scene_frame, auto_update=False)
        scene_layout.addWidget(self.viewport.interactor, 1)
        body.addWidget(scene_frame, 1)
        self.viewport.set_background("#edf2f6", top="#fafcfd")
        self.viewport.enable_anti_aliasing("fxaa")
        scene_key = QLabel('<span style="color:#d59420">●</span> Tool tip &nbsp; '
                           '<span style="color:#199a78">●</span> Target &nbsp; '
                           '<span style="color:#bc4046">●</span> Protected structure &nbsp; '
                           '<span style="color:#7757b5">━</span> Route')
        scene_key.setWordWrap(True)
        scene_key.setStyleSheet("background: white; padding: 8px 12px; font-size: 9pt;")
        scene_layout.addWidget(scene_key)

        self.position_label = QLabel()
        self.target_label = QLabel()
        self.clearance_label = QLabel()
        readouts = QHBoxLayout()
        for label in (self.position_label, self.target_label, self.clearance_label):
            label.setWordWrap(True)
            label.setStyleSheet("padding: 10px; background: white; border: 1px solid #d9e1e8;")
            readouts.addWidget(label, 1)
        layout.addLayout(readouts)
        self.message_label = QLabel("Ready for manual movement.")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        self.sensor_timer = QTimer(self)
        self.sensor_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.sensor_timer.timeout.connect(self.update_measurements)
        self.readout_timer = QTimer(self)
        self.readout_timer.timeout.connect(self.update_sensor_readouts)
        self.load_scenario(self.scenario_combo.currentIndex())
        self.scenario_combo.currentIndexChanged.connect(self.load_scenario)

        QApplication.instance().installEventFilter(self)
        self.last_tick = perf_counter()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_movement)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.start(16)

    def load_scenario(self, index):
        self.scenario_name = self.scenario_combo.itemText(index)
        scenario = SCENARIOS[self.scenario_name]
        self.configuration = scenario["start"].copy()
        self.target = tip_position(self.port, *scenario["target_configuration"])
        self.structure_centre = scenario["structure_centre"].copy()
        self.structure_radius = scenario["structure_radius"]
        # places the target on the near surface of the reference ellipsoid
        self.organ_centre = self.target + np.array([20.0, 0.0, 0.0])
        direction = (tip_position(self.port, *self.configuration) - self.port) / self.configuration[2]
        handle_end = self.port - (self.shaft_length - self.configuration[2] + 28) * direction
        reference_points = np.array([
            handle_end, self.port, tip_position(self.port, *self.configuration),
            self.organ_centre - [20, 26, 22], self.organ_centre + [20, 26, 22],
            self.structure_centre - self.structure_radius,
            self.structure_centre + self.structure_radius,
        ])
        self.scene_bounds = tuple(value for pair in zip(reference_points.min(axis=0) - 8,
                                                       reference_points.max(axis=0) + 8)
                                  for value in pair)
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
        self.viewport.enable_lightkit()
        organ = pv.Sphere(radius=1, theta_resolution=64, phi_resolution=48)
        organ.points = organ.points * np.array([20, 26, 22]) + self.organ_centre
        self.viewport.add_mesh(organ, color="#d9a59b", opacity=0.85,
                               smooth_shading=True, specular=0.25, specular_power=25)
        self.viewport.add_mesh(pv.Sphere(radius=self.structure_radius, center=self.structure_centre),
                               color="#bc4046", smooth_shading=True, specular=0.3)
        margin = pv.Sphere(radius=self.structure_radius + self.required_clearance,
                           center=self.structure_centre, theta_resolution=48, phi_resolution=32)
        self.viewport.add_mesh(margin, color="#ce7a7e", opacity=0.12, smooth_shading=True)
        self.viewport.add_mesh(pv.Sphere(radius=2.2, center=self.target),
                               color="#199a78", smooth_shading=True, ambient=0.25)
        self.viewport.add_mesh(pv.Disc(center=self.port, normal=(1, 0, 0),
                                       inner=3, outer=6, c_res=64),
                               color="#438198", ambient=0.35)

        # builds each instrument part once; movement changes its transform
        self.shaft_actor = self.viewport.add_mesh(
            pv.Cylinder(center=(0.5, 0, 0), direction=(1, 0, 0),
                        radius=self.tool_radius, height=1, resolution=32),
            color="#8797a5", smooth_shading=True, specular=0.7, specular_power=40)
        self.external_shaft_actor = self.viewport.add_mesh(
            pv.Cylinder(center=(-0.5, 0, 0), direction=(1, 0, 0),
                        radius=self.tool_radius, height=1, resolution=32),
            color="#8797a5", smooth_shading=True, specular=0.7, specular_power=40)
        grip = pv.Sphere(theta_resolution=48, phi_resolution=32)
        grip.points = grip.points * np.array([28, 8, 13]) + np.array([-14, 0, 0])
        self.handle_actor = self.viewport.add_mesh(grip, color="#286e9f",
                                                  smooth_shading=True, specular=0.35)
        self.collar_actor = self.viewport.add_mesh(
            pv.Cylinder(center=(-2, 0, 0), direction=(1, 0, 0), radius=2.4,
                        height=4, resolution=32), color="#526b80", smooth_shading=True)
        self.tip_actor = self.viewport.add_mesh(pv.Sphere(radius=self.tool_radius),
                                                color="#d59420", smooth_shading=True)

        # keeps sensor overlays distinct from the physical instrument
        marker = pv.PolyData(
            np.array([[-3.5, 0, 0], [3.5, 0, 0], [0, -3.5, 0],
                      [0, 3.5, 0], [0, 0, -3.5], [0, 0, 3.5]], dtype=float),
            lines=np.array([2, 0, 1, 2, 2, 3, 2, 4, 5]))
        self.tool_measurement_actor = self.viewport.add_mesh(
            marker, color="#287dc0", line_width=2, render_lines_as_tubes=True)
        self.target_measurement_actor = self.viewport.add_mesh(
            marker, color="#b050a2", line_width=2, render_lines_as_tubes=True)
        self.tool_estimate_actor = self.viewport.add_mesh(
            marker, color="#12536e", line_width=3, render_lines_as_tubes=True)
        self.target_estimate_actor = self.viewport.add_mesh(
            marker, color="#723d76", line_width=3, render_lines_as_tubes=True)
        for actor in (self.tool_measurement_actor, self.target_measurement_actor):
            actor.SetVisibility(self.measurements_checkbox.isChecked())
        for actor in (self.tool_estimate_actor, self.target_estimate_actor):
            actor.SetVisibility(self.estimates_checkbox.isChecked())
        self.label_actor = self.viewport.add_point_labels(
            np.array([self.port, self.target, self.structure_centre,
                      self.organ_centre + np.array([0.0, 0.0, 22.0])]),
            ["Fixed port", "Target", "Protected structure", "Organ surface"],
            font_size=12, text_color="#25384b", point_size=0, shape_opacity=0.0,
            always_visible=True, show_points=False)
        self.label_actor.SetVisibility(self.labels_checkbox.isChecked())
        self.viewport.add_axes(color="#62788c")
        self.reset_camera()

    def reset_camera(self):
        # fits fixed scenario bounds so noise + instrument resets cannot shift the view
        focal = (0, 0, 0)
        cameras = {
            "Perspective": [(-90, -260, 170), focal, (0, 0, 1)],
            "Top": [(0, 0, 300), focal, (0, 1, 0)],
            "Front": [(0, -300, 0), focal, (0, 0, 1)],
            "Overview": [(-90, -260, 170), focal, (0, 0, 1)],
        }
        self.viewport.camera_position = cameras[self.view_combo.currentText()]
        self.viewport.camera.parallel_projection = self.view_combo.currentText() in ("Top", "Front")
        low = np.minimum([-160, -115, -95], self.organ_centre - [20, 26, 22])
        high = np.maximum([135, 115, 95], self.organ_centre + [20, 26, 22])
        bounds = (tuple(value for pair in zip(low, high) for value in pair)
                  if self.view_combo.currentText() == "Overview" else self.scene_bounds)
        self.viewport.reset_camera(bounds=bounds)
        self.viewport.render()
        self.render_pending = False

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
            pv.lines_from_points(np.array(points)), color="#7757b5", line_width=3, render_lines_as_tubes=True)
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
        self.reset_measurements()
        self.message_label.setText("Instrument reset to a new random starting position. Plan a new route or use manual controls.")

    def reset_measurements(self):
        # restarts sensor noise separately from random instrument positions
        self.sensor_generator = np.random.default_rng(self.sensor_seed.value())
        self.measurement_count = 0
        self.tool_filter = None
        self.target_filter = None
        self.measurement_squared_error = np.zeros(2)
        self.estimate_squared_error = np.zeros(2)
        self.last_measurement_time = None
        self.sensor_timer.start(100)
        self.update_measurements()
        self.update_sensor_readouts()
        self.readout_timer.start(500)

    def update_measurements(self):
        sample_time = perf_counter()
        tool_tip = tip_position(self.port, *self.configuration)
        self.tool_measurement = measure_position(
            tool_tip, self.tool_noise.value(), self.sensor_generator)
        self.target_measurement = measure_position(
            self.target, self.target_noise.value(), self.sensor_generator)
        if self.tool_filter is None:
            # initialises each estimate from its first sensor reading
            self.tool_filter = KalmanFilter(self.tool_measurement, self.tool_noise.value(),
                                            acceleration_std=20.0)
            self.target_filter = KalmanFilter(self.target_measurement, self.target_noise.value(),
                                              acceleration_std=1.0)
        else:
            # predicts over the elapsed time between observations
            timestep = max(sample_time - self.last_measurement_time, 1e-6)
            for estimator, measurement in ((self.tool_filter, self.tool_measurement),
                                           (self.target_filter, self.target_measurement)):
                estimator.predict(timestep)
                estimator.update(measurement)
        self.last_measurement_time = sample_time
        self.measurement_count += 1
        self.tool_measurement_actor.SetPosition(*self.tool_measurement)
        self.target_measurement_actor.SetPosition(*self.target_measurement)
        self.tool_estimate_actor.SetPosition(*self.tool_filter.state[:3])
        self.target_estimate_actor.SetPosition(*self.target_filter.state[:3])
        # records errors at the same instant as each sensor sample
        self.measurement_errors = np.array([
            np.linalg.norm(self.tool_measurement - tool_tip),
            np.linalg.norm(self.target_measurement - self.target)])
        self.estimate_errors = np.array([
            np.linalg.norm(self.tool_filter.state[:3] - tool_tip),
            np.linalg.norm(self.target_filter.state[:3] - self.target)])
        self.measurement_squared_error += self.measurement_errors ** 2
        self.estimate_squared_error += self.estimate_errors ** 2
        if self.measurements_checkbox.isChecked() or self.estimates_checkbox.isChecked():
            self.render_pending = True

    def update_sensor_readouts(self):
        # refreshes one snapshot while sampling + filtering continue at 10 Hz
        readings = [
            (self.tool_measurement_label, "Tool tip", self.tool_measurement),
            (self.target_measurement_label, "Target", self.target_measurement),
            (self.tool_estimate_label, "Tool tip", self.tool_filter.state[:3]),
            (self.target_estimate_label, "Target", self.target_filter.state[:3]),
        ]
        for label, name, position in readings:
            headings = "".join(f'<td width="33%" align="right">{axis}</td>' for axis in "XYZ")
            values = "".join(f'<td align="right">{value:.2f}</td>' for value in position)
            label.setText(f'<b>{name}</b><table width="100%" cellspacing="4">'
                          f'<tr style="color:#62788c">{headings}</tr>'
                          f'<tr style="font-family:Consolas">{values}</tr></table>')
        self.measurement_error_label.setText(
            f"Raw error · mm\nTool {self.measurement_errors[0]:.2f}   Target {self.measurement_errors[1]:.2f}")
        self.estimate_error_label.setText(
            f"Filtered error · mm\nTool {self.estimate_errors[0]:.2f}   Target {self.estimate_errors[1]:.2f}")
        # calculates model RMS uncertainty from the three position variances
        tool_uncertainty = np.sqrt(max(0.0, np.trace(self.tool_filter.covariance[:3, :3])))
        target_uncertainty = np.sqrt(max(0.0, np.trace(self.target_filter.covariance[:3, :3])))
        self.uncertainty_label.setText(
            f"Position uncertainty · mm\nTool {tool_uncertainty:.2f}   Target {target_uncertainty:.2f}")
        measured_rmse = np.sqrt(self.measurement_squared_error / self.measurement_count)
        filtered_rmse = np.sqrt(self.estimate_squared_error / self.measurement_count)
        self.rmse_label.setText(
            '<b>Running 3D RMSE · mm</b><table width="100%" cellspacing="4">'
            '<tr><td></td><td align="right">Raw</td><td align="right">Filtered</td></tr>'
            f'<tr><td>Tool</td><td align="right">{measured_rmse[0]:.2f}</td>'
            f'<td align="right">{filtered_rmse[0]:.2f}</td></tr>'
            f'<tr><td>Target</td><td align="right">{measured_rmse[1]:.2f}</td>'
            f'<td align="right">{filtered_rmse[1]:.2f}</td></tr></table>')
        self.displayed_measurement_count = self.measurement_count
        self.measurement_count_label.setText(f"Sample {self.displayed_measurement_count}  ·  Readouts 2 Hz")
        tip = tip_position(self.port, *self.configuration)
        target_distance = np.linalg.norm(tip - self.target)
        clearance = shaft_clearance(self.port, tip, self.tool_radius,
                                    self.structure_centre, self.structure_radius)
        yaw, pitch = np.rad2deg(self.configuration[:2])
        self.position_label.setText(f"ACTUAL INSTRUMENT\nYaw {yaw:.1f}° · Pitch {pitch:.1f}° · Depth {self.configuration[2]:.1f} mm")
        self.target_label.setText(f"TIP TO TARGET\n{target_distance:.2f} mm")
        self.clearance_label.setText(f"SHAFT CLEARANCE\n{clearance:.2f} mm  ·  Required {self.required_clearance:.1f} mm")

    def toggle_measurements(self, _visible):
        for actor in (self.tool_measurement_actor, self.target_measurement_actor):
            actor.SetVisibility(self.measurements_checkbox.isChecked())
        for actor in (self.tool_estimate_actor, self.target_estimate_actor):
            actor.SetVisibility(self.estimates_checkbox.isChecked())
        self.viewport.render()

    def toggle_labels(self, visible):
        self.label_actor.SetVisibility(visible)
        self.viewport.render()

    def advance_movement(self):
        try:
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
        finally:
            if self.render_pending:
                self.viewport.render()
                self.render_pending = False

    def update_instrument(self):
        yaw, pitch, depth = self.configuration
        # rotates local +X onto the instrument direction
        rotation = np.array([
            [np.cos(pitch) * np.cos(yaw), -np.sin(yaw), -np.sin(pitch) * np.cos(yaw)],
            [np.cos(pitch) * np.sin(yaw), np.cos(yaw), -np.sin(pitch) * np.sin(yaw)],
            [np.sin(pitch), 0, np.cos(pitch)],
        ])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation @ np.diag([depth, 1, 1])
        matrix[:3, 3] = self.port
        self.shaft_actor.user_matrix = matrix
        # moves the external handle towards the port as insertion increases
        external_length = self.shaft_length - depth
        matrix[:3, :3] = rotation @ np.diag([external_length, 1, 1])
        self.external_shaft_actor.user_matrix = matrix
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.port - external_length * rotation[:, 0]
        self.handle_actor.user_matrix = matrix
        self.collar_actor.user_matrix = matrix
        self.tip_actor.SetPosition(*tip_position(self.port, *self.configuration))
        self.render_pending = True

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
        self.readout_timer.stop()
        if self.planning_cancel is not None:
            self.planning_cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
        QApplication.instance().removeEventFilter(self)
        self.viewport.close()
        super().closeEvent(event)
