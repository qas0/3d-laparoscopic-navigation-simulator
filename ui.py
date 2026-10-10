from concurrent.futures import ThreadPoolExecutor
from collections import deque
from queue import Queue
from threading import Event
from time import perf_counter

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from astar import find_path, get_uncertainty_margin
from experiments import CompareMethods
from feedback import get_target_feedback
from geometry import movement_is_clear, shaft_clearance, tip_position
from kalman import KalmanFilter
from motion import GetRespiratoryMotion
from scenarios import SCENARIOS
from sensors import ApplyBias, ApplyDropout, measure_position


def FormatReading(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}"


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
        self.route_clearance = self.required_clearance
        self.lower_limits = np.array([np.deg2rad(-45), np.deg2rad(-35), 10.0])
        self.upper_limits = np.array([np.deg2rad(45), np.deg2rad(35), 110.0])
        self.speed_limits = np.array([np.deg2rad(20), np.deg2rad(20), 15.0])
        self.target_tolerance = 2.0
        self.arrival_dwell = 0.5
        self.tracking_timeout = 0.25
        self.route_feedback_active = False
        self.route_finished_time = None
        self.route_result = None
        self.arrival_started = [None, None, None]
        self.arrival_confirmed = [False, False, False]
        self.random_generator = np.random.default_rng()
        self.path = None
        self.path_actor = None
        self.route_points = None
        self.manual_guidance = False
        self.guidance_warning = None
        self.guidance_warning_pending = None
        self.guidance_warning_started = None
        self.margin_actor = None
        self.path_index = 0
        self.following = False
        self.planning_future = None
        self.planning_cancel = None
        self.comparison_future = None
        self.comparison_cancel = Event()
        self.comparison_progress = Queue()
        self.comparison_result = None
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
            QFrame#controls, QFrame#scene { background: white;
                        border: 1px solid #d9e1e8; border-radius: 8px; }
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
            QTabBar::tab { background: #e8eef3; padding: 9px 8px; font-size: 9pt; }
            QTabBar::tab:selected { background: white; color: #286e9f; }
            QTableWidget { background: white; alternate-background-color: #edf2f6;
                           gridline-color: #d9e1e8; }
            QHeaderView::section { background: #e8eef3; padding: 7px; border: none; }
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
        body = QHBoxLayout()
        layout.addLayout(body, 1)

        self.control_tabs = QTabWidget()
        self.control_tabs.setFixedWidth(330)
        body.addWidget(self.control_tabs)
        control_layout = self.create_control_tab("Navigation")

        scenario_title = QLabel("Scenario")
        scenario_title.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(scenario_title)
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(list(SCENARIOS))
        control_layout.addWidget(self.scenario_combo)
        self.scenario_description = QLabel(wordWrap=True)
        control_layout.addWidget(self.scenario_description)
        motion_group = QGroupBox("Respiratory motion")
        motion_settings = QFormLayout(motion_group)
        self.motion_amplitude = QDoubleSpinBox(minimum=0.0, maximum=10.0, decimals=1,
                                              singleStep=0.5, suffix=" mm", keyboardTracking=False)
        self.motion_frequency = QDoubleSpinBox(minimum=0.0, maximum=0.5, decimals=2,
                                              singleStep=0.05, value=0.2, suffix=" Hz",
                                              keyboardTracking=False)
        self.motion_amplitude.setToolTip("Maximum Z displacement from the resting position. Zero keeps anatomy still.")
        self.motion_frequency.setToolTip("Cycles per second: 0.2 Hz gives one cycle every 5 seconds.")
        motion_group.setToolTip("Moves organ + target together along Z. Changing settings restarts the selected scenario.")
        motion_settings.addRow("Amplitude", self.motion_amplitude)
        motion_settings.addRow("Frequency", self.motion_frequency)
        self.motion_label = QLabel(wordWrap=True)
        motion_settings.addRow(self.motion_label)
        control_layout.addWidget(motion_group)
        control_layout.addWidget(QLabel("Guidance source"))
        self.navigation_mode_combo = QComboBox()
        self.navigation_mode_combo.addItems(["Ideal positions", "Raw tracking", "Filtered tracking"])
        self.navigation_mode_combo.setCurrentIndex(2)
        self.navigation_mode_combo.setToolTip("Plans towards a target snapshot from this source. Changing source cancels the route.")
        control_layout.addWidget(self.navigation_mode_combo)

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
            slider = QSlider(Qt.Orientation.Horizontal, minimum=minimum * 10,
                             maximum=maximum * 10, value=value * 10)
            slider.setToolTip(f"Command {name.lower()}; actual position appears below.")
            slider.valueChanged.connect(lambda _, axis=index: self.set_command(axis))
            instrument_layout.addWidget(label)
            instrument_layout.addWidget(slider)
            self.command_labels.append(label)
            self.sliders.append(slider)
        control_layout.addWidget(instrument_group)

        planning_group = QGroupBox("A* navigation")
        planning_layout = QVBoxLayout(planning_group)
        self.planning_mode_combo = QComboBox()
        self.planning_mode_combo.addItem("Conventional A*", "conventional")
        self.planning_mode_combo.addItem("Proximity-aware A*", "prox_aware")
        self.planning_mode_combo.addItem("Uncertainty-aware A*", "uncert_aware")
        self.planning_mode_combo.setToolTip("All methods enforce shaft clearance. Uncertainty mode adds a frozen tool-tracking margin. Changing method cancels the route.")
        self.proximity_weight = QDoubleSpinBox(minimum=0.0, maximum=10.0, singleStep=0.5,
                                              decimals=1, value=1.0, enabled=False)
        self.proximity_weight.setToolTip("Zero uses travel cost only. Higher weights discourage close passes; the penalty decays over 5 mm above the base clearance.")
        self.uncertainty_scale = QDoubleSpinBox(minimum=0.0, maximum=5.0, singleStep=0.5,
                                               decimals=1, value=1.0, enabled=False)
        self.uncertainty_scale.setToolTip("Adds filtered tool-tip disagreement + k times the largest-direction standard deviation. Zero k keeps the disagreement term. k is an experimental sensitivity, not a confidence percentage.")
        planning_settings = QFormLayout()
        planning_settings.setHorizontalSpacing(4)
        planning_settings.addRow("Planning method", self.planning_mode_combo)
        planning_settings.addRow("Proximity weight", self.proximity_weight)
        planning_settings.addRow("Uncertainty k", self.uncertainty_scale)
        planning_layout.addLayout(planning_settings)
        self.plan_button = QPushButton("Plan route", objectName="primary")
        self.plan_button.clicked.connect(self.plan_path)
        self.follow_button = QPushButton("Follow route", enabled=False)
        self.follow_button.setToolTip("Automatically executes a freshly planned route for comparison. Manual controls keep the route as a reference.")
        self.follow_button.clicked.connect(self.follow_path)
        self.cancel_button = QPushButton("Cancel route", enabled=False)
        self.cancel_button.clicked.connect(self.stop_path)
        route_buttons = QHBoxLayout()
        route_buttons.addWidget(self.plan_button)
        route_buttons.addWidget(self.follow_button)
        planning_layout.addLayout(route_buttons)
        planning_layout.addWidget(self.cancel_button)
        self.plan_label = QLabel("Grid: 5° yaw / pitch + 5 mm insertion\nTarget tolerance: 2 mm",
                                 wordWrap=True, minimumHeight=52)
        planning_layout.addWidget(self.plan_label)
        self.margin_label = QLabel(wordWrap=True, visible=False)
        self.margin_label.setToolTip("Uses filtered tool tracking independently of target guidance. The fixed-port model assumes the frozen endpoint-error bound applies throughout this route. Static known geometry and configuration remain the baseline; tracking changes do not update this margin automatically.")
        planning_layout.addWidget(self.margin_label)
        self.route_result_label = QLabel(wordWrap=True, minimumHeight=50, visible=False)
        self.route_result_label.setToolTip("Tip-to-guide distance uses the selected tracking source + the displayed tip polyline. It does not check shaft clearance or prove that the instrument follows the planned configurations. Tracking changes must persist for 0.5 s before a replan warning stays visible.")
        planning_layout.addWidget(self.route_result_label)
        control_layout.addWidget(planning_group)

        keyboard_help = QLabel("W / S  Insert / retract\n"
                               "A / D  Yaw       Q / E  Pitch\n"
                               "Hold a key to move. Manual control keeps the route as a guide.",
                               wordWrap=True)
        control_layout.addWidget(keyboard_help)

        self.pause_button = QPushButton("Pause movement", checkable=True)
        self.pause_button.setToolTip("Pauses the instrument; anatomy + sensors continue.")
        self.pause_button.toggled.connect(self.set_paused)
        control_layout.addWidget(self.pause_button)
        self.reset_button = QPushButton("Reset instrument")
        self.reset_button.setToolTip("Places the instrument at a different random starting position.")
        self.reset_button.clicked.connect(self.reset_instrument)
        control_layout.addWidget(self.reset_button)
        control_layout.addStretch()
        scene_note = QLabel("Tool radius: 1.5 mm\nRequired shaft clearance: 2.0 mm", wordWrap=True)
        control_layout.addWidget(scene_note)

        sensor_layout = self.create_control_tab("Sensors")

        sensor_group = QGroupBox("Position measurements")
        sensor_settings = QFormLayout(sensor_group)
        self.tool_noise = QDoubleSpinBox(minimum=0.0, maximum=10.0, decimals=1,
                                        singleStep=0.1, suffix=" mm", keyboardTracking=False)
        self.target_noise = QDoubleSpinBox(minimum=0.0, maximum=10.0, decimals=1,
                                          singleStep=0.1, suffix=" mm", keyboardTracking=False)
        for setting, value in [(self.tool_noise, 1.0), (self.target_noise, 2.0)]:
            setting.setValue(value)
            setting.setToolTip("Gaussian noise standard deviation for each coordinate.")
        sensor_settings.addRow("Tool noise σ", self.tool_noise)
        sensor_settings.addRow("Target noise σ", self.target_noise)
        self.tool_dropout = QDoubleSpinBox(minimum=0.0, maximum=100.0, decimals=0,
                                          singleStep=5, suffix=" %", keyboardTracking=False)
        self.target_dropout = QDoubleSpinBox(minimum=0.0, maximum=100.0, decimals=0,
                                            singleStep=5, suffix=" %", keyboardTracking=False)
        for name, setting in (("Tool dropout", self.tool_dropout),
                              ("Target dropout", self.target_dropout)):
            setting.setToolTip("Probability of discarding each reading. Changes preserve the filter.")
            sensor_settings.addRow(name, setting)
        self.sensor_seed = QSpinBox(minimum=0, maximum=999999, value=42, keyboardTracking=False)
        self.sensor_seed.setToolTip("Repeats Gaussian noise + dropout sequences.")
        sensor_settings.addRow("Sensor seed", self.sensor_seed)
        sensor_layout.addWidget(sensor_group)
        bias_group = QGroupBox("Sensor bias + drift")
        bias_settings = QFormLayout(bias_group)
        self.sensor_axis = QComboBox()
        self.sensor_axis.addItems(["X", "Y", "Z"])
        self.sensor_axis.setToolTip("World axis for fixed bias + subsequent drift increments.")
        bias_settings.addRow("Direction", self.sensor_axis)
        self.tool_bias = QDoubleSpinBox(minimum=-20.0, maximum=20.0, decimals=1,
                                       singleStep=0.5, suffix=" mm", keyboardTracking=False)
        self.target_bias = QDoubleSpinBox(minimum=-20.0, maximum=20.0, decimals=1,
                                         singleStep=0.5, suffix=" mm", keyboardTracking=False)
        self.tool_drift = QDoubleSpinBox(minimum=-1.0, maximum=1.0, decimals=2,
                                        singleStep=0.05, suffix=" mm/s", keyboardTracking=False)
        self.target_drift = QDoubleSpinBox(minimum=-1.0, maximum=1.0, decimals=2,
                                          singleStep=0.05, suffix=" mm/s", keyboardTracking=False)
        for name, setting in (("Tool bias", self.tool_bias), ("Target bias", self.target_bias),
                              ("Tool drift", self.tool_drift), ("Target drift", self.target_drift)):
            bias_settings.addRow(name, setting)
        bias_group.setToolTip("Changes preserve the filters. Drift accumulates until Restart sensors clears it; zero rate stops further drift.")
        sensor_layout.addWidget(bias_group)
        self.restart_sensor_button = QPushButton("Restart sensors")
        self.restart_sensor_button.clicked.connect(self.reset_measurements)
        sensor_layout.addWidget(self.restart_sensor_button)
        self.measurements_checkbox = QCheckBox("Raw positions in scene", checked=False)
        self.measurements_checkbox.setToolTip("Raw observations: blue tool + pink target. These markers are not physical objects.")
        self.measurements_checkbox.toggled.connect(self.toggle_measurements)
        sensor_layout.addWidget(self.measurements_checkbox)
        self.estimates_checkbox = QCheckBox("Filtered positions in scene", checked=False)
        self.estimates_checkbox.setToolTip("Filtered crosses: navy tool + purple target. Their size does not represent uncertainty.")
        self.estimates_checkbox.toggled.connect(self.toggle_measurements)
        sensor_layout.addWidget(self.estimates_checkbox)
        sensor_note = QLabel("Sensors: 10 Hz  ·  Readouts: 2 Hz", wordWrap=True)
        sensor_layout.addWidget(sensor_note)
        measurements_group = QGroupBox("Raw position readings · mm")
        measurements_layout = QVBoxLayout(measurements_group)
        self.tool_measurement_label = QLabel(wordWrap=True)
        self.target_measurement_label = QLabel(wordWrap=True)
        self.measurement_error_label = QLabel(wordWrap=True)
        for label in (self.tool_measurement_label, self.target_measurement_label,
                      self.measurement_error_label):
            measurements_layout.addWidget(label)
        sensor_layout.addWidget(measurements_group)

        estimate_layout = self.create_control_tab("Feedback")
        feedback_group = QGroupBox("Target distance + arrival")
        feedback_layout = QVBoxLayout(feedback_group)
        self.feedback_distance_label = QLabel(wordWrap=True)
        self.feedback_status_label = QLabel(wordWrap=True, minimumHeight=42)
        self.relative_uncertainty_label = QLabel(wordWrap=True)
        self.relative_uncertainty_label.setToolTip("RMS uncertainty in the relative tool-target position, assuming independent filter errors. It is not a 95% radius or an arrival probability.")
        for label in (self.feedback_distance_label, self.feedback_status_label,
                      self.relative_uncertainty_label):
            feedback_layout.addWidget(label)
        estimate_layout.addWidget(feedback_group)

        figure = Figure(figsize=(3, 1.8), dpi=100, facecolor="white")
        self.distance_canvas = FigureCanvasQTAgg(figure)
        self.distance_canvas.setFixedHeight(180)
        self.distance_axes = figure.add_subplot(111)
        figure.subplots_adjust(left=0.19, right=0.96, bottom=0.25, top=0.95)
        self.distance_axes.set_xlabel("Time (s)", fontsize=8)
        self.distance_axes.set_ylabel("Distance (mm)", fontsize=8)
        self.distance_axes.tick_params(labelsize=8)
        self.distance_axes.grid(alpha=0.2)
        self.distance_lines = [self.distance_axes.plot([], [], color=color, linewidth=width)[0]
                               for color, width in [("#667584", 1.3), ("#b8cad9", 1.0), ("#286e9f", 1.8)]]
        self.distance_axes.axhline(self.target_tolerance, color="#bc4046", linestyle="--", linewidth=1)
        estimate_layout.addWidget(self.distance_canvas)
        chart_key = QLabel('<span style="color:#667584">━</span> Actual &nbsp; '
                           '<span style="color:#94afc4">━</span> Raw &nbsp; '
                           '<span style="color:#286e9f">━</span> Filtered<br>'
                           '<span style="color:#bc4046">┄</span> 2 mm arrival tolerance · Last 30 s',
                           wordWrap=True)
        chart_key.setStyleSheet("font-size: 9pt;")
        estimate_layout.addWidget(chart_key)

        details_button = QPushButton("Position + filter details", checkable=True)
        estimate_layout.addWidget(details_button)
        filter_group = QGroupBox("Filtered position readings · mm", visible=False)
        filter_layout = QVBoxLayout(filter_group)
        self.tool_estimate_label = QLabel(wordWrap=True)
        self.target_estimate_label = QLabel(wordWrap=True)
        self.estimate_error_label = QLabel(wordWrap=True)
        self.uncertainty_label = QLabel(wordWrap=True)
        self.rmse_label = QLabel(wordWrap=True)
        self.uncertainty_label.setToolTip("Model RMS position uncertainty: square root of the three position variances added together. Unknown bias + drift are not included.")
        self.rmse_label.setToolTip("Raw RMSE uses received readings only. Filtered RMSE uses every initialized estimate, including predictions during dropout. These are different sample sets.")
        for label in (self.tool_estimate_label, self.target_estimate_label,
                      self.estimate_error_label, self.uncertainty_label, self.rmse_label):
            filter_layout.addWidget(label)
        estimate_layout.addWidget(filter_group)
        details_button.toggled.connect(filter_group.setVisible)
        self.measurement_count_label = QLabel()
        sensor_layout.addWidget(self.measurement_count_label)
        sensor_layout.addStretch()
        estimate_layout.addStretch()
        marker_note = QLabel("Raw: blue tool / pink target\n"
                             "Filtered: navy tool / purple target", wordWrap=True)
        sensor_layout.addWidget(marker_note)
        sensor_baseline = QLabel("Constant-velocity filtering.\n"
                                "Motion σa: tool 20 / target 1 mm/s².\n"
                                "Errors use true positions for evaluation.\n"
                                "Motion uses known configuration + obstacle geometry.\n"
                                "The selected tracking source supplies target + guide/arrival feedback.",
                                wordWrap=True)
        estimate_layout.addWidget(sensor_baseline)
        for setting in (self.tool_noise, self.target_noise, self.sensor_seed):
            setting.valueChanged.connect(self.reset_measurements)

        comparison_layout = self.create_control_tab("Experiments")
        self.comparison_setup_label = QLabel(wordWrap=True)
        comparison_layout.addWidget(self.comparison_setup_label)
        comparison_settings = QFormLayout()
        self.comparison_trials = QSpinBox(minimum=1, maximum=100, value=10,
                                         keyboardTracking=False)
        comparison_settings.addRow("Trials per method", self.comparison_trials)
        comparison_layout.addLayout(comparison_settings)
        self.comparison_button = QPushButton("Run comparison", objectName="primary")
        self.comparison_button.setToolTip("Runs all three methods with the selected settings, starting at the scenario's fixed pose. Each trial restarts sensors + breathing, uses a 1 s warmup and a 20 s simulation limit.")
        self.comparison_button.clicked.connect(self.RunComparison)
        comparison_layout.addWidget(self.comparison_button)
        self.comparison_stop_button = QPushButton("Stop comparison", enabled=False)
        self.comparison_stop_button.clicked.connect(self.comparison_cancel.set)
        comparison_layout.addWidget(self.comparison_stop_button)
        self.comparison_status_label = QLabel(wordWrap=True)
        comparison_layout.addWidget(self.comparison_status_label)
        self.comparison_dialog = QDialog(self)
        self.comparison_dialog.setWindowTitle("Planner comparison")
        self.comparison_dialog.resize(1020, 650)
        self.comparison_dialog.setMinimumSize(820, 450)
        result_layout = QVBoxLayout(self.comparison_dialog)
        self.comparison_result_label = QLabel(wordWrap=True)
        result_layout.addWidget(self.comparison_result_label)
        self.comparison_table = QTableWidget(0, 4)
        self.comparison_table.setToolTip(
            "Mean ± sample SD across independent trials; n counts trials.\n"
            "Predicted RMS = sqrt(mean(trace(P))) over the filtered RMSE samples.\n"
            "Coverage counts errors inside the nominal 95% covariance region, including warmup + dropout predictions.\n"
            "Trials can stop early, so coverage describes their observed samples.")
        self.comparison_table.setHorizontalHeaderLabels(
            ["Metric / mean ± SD (n)", "Conventional A*", "Proximity-aware A*", "Uncertainty-aware A*"])
        self.comparison_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.verticalHeader().hide()
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        result_layout.addWidget(self.comparison_table)
        self.comparison_results_button = QPushButton("View results", visible=False)
        self.comparison_results_button.clicked.connect(self.comparison_dialog.show)
        comparison_layout.addWidget(self.comparison_results_button)
        comparison_layout.addStretch()

        scene_frame = QFrame(objectName="scene")
        scene_layout = QVBoxLayout(scene_frame)
        scene_layout.setContentsMargins(1, 1, 1, 1)
        scene_layout.setSpacing(0)
        scene_toolbar = QHBoxLayout()
        scene_toolbar.setContentsMargins(12, 8, 12, 8)
        scene_title = QLabel("3D WORKSPACE")
        scene_title.setStyleSheet("font-size: 9pt; font-weight: 600; color: #62788c;")
        scene_toolbar.addWidget(scene_title)
        scene_toolbar.addStretch()
        self.labels_checkbox = QCheckBox("Labels", checked=False)
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
                           '<span style="color:#7757b5">━</span> Route', wordWrap=True)
        scene_key.setStyleSheet("background: white; padding: 8px 12px; font-size: 9pt;")
        scene_layout.addWidget(scene_key)

        self.position_label = QLabel(wordWrap=True)
        self.target_label = QLabel(wordWrap=True)
        self.clearance_label = QLabel(wordWrap=True)
        readouts = QHBoxLayout()
        for label in (self.position_label, self.target_label, self.clearance_label):
            label.setStyleSheet("padding: 10px; background: white; border: 1px solid #d9e1e8;")
            readouts.addWidget(label, 1)
        layout.addLayout(readouts)
        self.message_label = QLabel("Ready for manual movement.", wordWrap=True)
        layout.addWidget(self.message_label)
        self.sensor_timer = QTimer(self)
        self.sensor_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.sensor_timer.timeout.connect(self.update_measurements)
        self.readout_timer = QTimer(self)
        self.readout_timer.timeout.connect(self.update_sensor_readouts)
        self.readout_timer.timeout.connect(self.UpdateComparison)
        self.load_scenario(self.scenario_combo.currentIndex())
        self.scenario_combo.currentIndexChanged.connect(self.load_scenario)
        for setting in (self.motion_amplitude, self.motion_frequency):
            setting.valueChanged.connect(lambda _: self.load_scenario(self.scenario_combo.currentIndex()))
        self.navigation_mode_combo.currentIndexChanged.connect(self.stop_path)
        self.navigation_mode_combo.currentIndexChanged.connect(self.update_sensor_readouts)
        self.planning_mode_combo.currentIndexChanged.connect(self.stop_path)
        self.planning_mode_combo.currentIndexChanged.connect(
            lambda index: self.proximity_weight.setEnabled(index in (1, 2)))
        self.planning_mode_combo.currentIndexChanged.connect(
            lambda index: self.uncertainty_scale.setEnabled(index == 2))
        self.proximity_weight.valueChanged.connect(self.stop_path)
        self.uncertainty_scale.valueChanged.connect(self.stop_path)
        self.control_tabs.currentChanged.connect(self.update_sensor_readouts)
        self.control_tabs.currentChanged.connect(self.UpdateComparison)
        self.UpdateComparison()

        QApplication.instance().installEventFilter(self)
        self.last_tick = perf_counter()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_movement)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.start(16)

    def create_control_tab(self, title):
        panel = QFrame(objectName="controls")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        self.control_tabs.addTab(scroll, title)
        return layout

    def load_scenario(self, index):
        self.scenario_name = self.scenario_combo.itemText(index)
        scenario = SCENARIOS[self.scenario_name]
        self.configuration = scenario["start"].copy()
        self.rest_target = tip_position(self.port, *scenario["target_configuration"])
        self.target = self.rest_target.copy()
        self.motion_started_time = None
        self.structure_centre = scenario["structure_centre"].copy()
        self.structure_radius = scenario["structure_radius"]
        # places the target on the near surface of the reference ellipsoid
        self.rest_organ_centre = self.rest_target + np.array([20.0, 0.0, 0.0])
        self.organ_centre = self.rest_organ_centre.copy()
        motion_bounds = np.array([0.0, 0.0, self.motion_amplitude.value()])
        direction = (tip_position(self.port, *self.configuration) - self.port) / self.configuration[2]
        handle_end = self.port - (self.shaft_length - self.configuration[2] + 28) * direction
        reference_points = np.array([
            handle_end, self.port, tip_position(self.port, *self.configuration),
            self.organ_centre - [20, 26, 22] - motion_bounds,
            self.organ_centre + [20, 26, 22] + motion_bounds,
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
        self.organ_actor = self.viewport.add_mesh(
            organ, color="#d9a59b", opacity=0.85,
            smooth_shading=True, specular=0.25, specular_power=25)
        self.viewport.add_mesh(pv.Sphere(radius=self.structure_radius, center=self.structure_centre),
                               color="#bc4046", smooth_shading=True, specular=0.3)
        margin = pv.Sphere(radius=self.structure_radius + self.required_clearance,
                           center=self.structure_centre, theta_resolution=48, phi_resolution=32)
        self.margin_actor = self.viewport.add_mesh(
            margin, color="#ce7a7e", opacity=0.12, smooth_shading=True)
        self.margin_actor.SetOrigin(*self.structure_centre)
        self.target_actor = self.viewport.add_mesh(
            pv.Sphere(radius=2.2, center=self.target),
            color="#199a78", smooth_shading=True, ambient=0.25)
        self.viewport.add_mesh(pv.Disc(center=self.port, normal=(1, 0, 0),
                                       inner=3, outer=6, c_res=64),
                               color="#438198", ambient=0.35)

        # builds each instrument part once
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
        (self.tool_measurement_actor, self.target_measurement_actor,
         self.tool_estimate_actor, self.target_estimate_actor) = [
            self.viewport.add_mesh(marker, color=color, line_width=width, render_lines_as_tubes=True)
            for color, width in [("#287dc0", 2), ("#b050a2", 2), ("#12536e", 3), ("#723d76", 3)]]
        for actor in (self.tool_measurement_actor, self.target_measurement_actor):
            actor.SetVisibility(self.measurements_checkbox.isChecked())
        for actor in (self.tool_estimate_actor, self.target_estimate_actor):
            actor.SetVisibility(self.estimates_checkbox.isChecked())
        self.label_points = pv.PolyData(np.array([
            self.port, self.target, self.structure_centre,
            self.organ_centre + np.array([0.0, 0.0, 22.0])]))
        self.label_points["labels"] = ["Fixed port", "Target", "Protected structure", "Organ surface"]
        self.label_actor = self.viewport.add_point_labels(
            self.label_points, "labels",
            font_size=12, text_color="#25384b", point_size=0, shape_opacity=0.0,
            always_visible=True, show_points=False)
        self.label_actor.SetVisibility(self.labels_checkbox.isChecked())
        self.viewport.add_axes(color="#62788c")
        self.reset_camera()

    def UpdateRespiratoryMotion(self, now=None):
        now = perf_counter() if now is None else now
        if self.motion_started_time is None:
            self.motion_started_time = now
        self.motion_displacement, self.motion_velocity = GetRespiratoryMotion(
            self.motion_amplitude.value(), self.motion_frequency.value(), now - self.motion_started_time)
        offset = np.array([0.0, 0.0, self.motion_displacement])
        target = self.rest_target + offset
        if np.array_equal(target, self.target):
            return
        self.target = target
        self.organ_centre = self.rest_organ_centre + offset
        # translates persistent meshes and labels without changing camera framing
        self.organ_actor.SetPosition(*offset)
        self.target_actor.SetPosition(*offset)
        self.label_points.points[1] = self.target
        self.label_points.points[3] = self.organ_centre + [0.0, 0.0, 22.0]
        self.render_pending = True

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

    def RunComparison(self):
        if self.comparison_future is not None:
            return
        if self.planning_future is not None:
            self.comparison_status_label.setText("Finish or cancel route planning first.")
            return
        self.UpdateComparison()
        # captures widget values before sending plain settings to the worker
        axis = np.eye(3)[self.sensor_axis.currentIndex()]
        settings = {
            "tracking": ("ideal", "raw", "filtered")[self.navigation_mode_combo.currentIndex()],
            "tool_noise": self.tool_noise.value(), "target_noise": self.target_noise.value(),
            "tool_dropout": self.tool_dropout.value() / 100,
            "target_dropout": self.target_dropout.value() / 100,
            "tool_bias": axis * self.tool_bias.value(), "target_bias": axis * self.target_bias.value(),
            "tool_drift": axis * self.tool_drift.value(), "target_drift": axis * self.target_drift.value(),
            "motion_amplitude": self.motion_amplitude.value(),
            "motion_frequency": self.motion_frequency.value(),
            "proximity_weight": self.proximity_weight.value(),
            "uncertainty_scale": self.uncertainty_scale.value(),
        }
        seed = self.sensor_seed.value()
        seeds = range(seed, seed + self.comparison_trials.value())
        self.comparison_cancel.clear()
        self.comparison_result = None
        while not self.comparison_progress.empty():
            self.comparison_progress.get_nowait()
        self.comparison_dialog.hide()
        self.comparison_results_button.hide()
        self.comparison_status_label.setText(f"Completed 0 / {3 * len(seeds)} trials.")
        self.comparison_future = self.executor.submit(
            CompareMethods, self.scenario_combo.currentText(), seeds,
            cancel_event=self.comparison_cancel, progress=self.comparison_progress.put, **settings)
        self.comparison_button.setEnabled(False)
        self.comparison_trials.setEnabled(False)
        self.comparison_stop_button.setEnabled(True)
        self.plan_button.setEnabled(False)

    def UpdateComparison(self):
        if self.comparison_future is None:
            seed, count = self.sensor_seed.value(), self.comparison_trials.value()
            self.comparison_setup_label.setText(
                f"{self.scenario_combo.currentText()}\n{self.navigation_mode_combo.currentText()}\n"
                f"3 methods × {count} trials · Seeds {seed}–{seed + count - 1}")
            return
        # reads worker progress on the UI thread without touching the live scene
        while not self.comparison_progress.empty():
            completed, total, mode, seed = self.comparison_progress.get_nowait()
            state = "Stopping" if self.comparison_cancel.is_set() else "Completed"
            self.comparison_status_label.setText(f"{state} · {completed} / {total} trials.")
        if not self.comparison_future.done():
            return
        future = self.comparison_future
        self.comparison_future = None
        self.comparison_button.setEnabled(True)
        self.comparison_trials.setEnabled(True)
        self.comparison_stop_button.setEnabled(False)
        self.plan_button.setEnabled(not self.route_feedback_active)
        try:
            result = future.result()
        except Exception as error:
            self.comparison_status_label.setText(f"Comparison failed: {error}")
            return
        self.comparison_result = result
        state = "Stopped" if result["cancelled"] else "Completed"
        self.comparison_status_label.setText(
            f"{state} · {len(result['trials'])} / {3 * len(result['seeds'])} trials.")
        if not result["trials"]:
            return
        settings = result["settings"]
        self.comparison_result_label.setText(
            f"{result['scenario']} · {settings['tracking'].capitalize()} tracking · "
            f"Seeds {result['seeds'][0]}–{result['seeds'][-1]}\n"
            f"Noise σ: tool {settings['tool_noise']:g} / target {settings['target_noise']:g} mm · "
            f"Dropout: tool {100 * settings['tool_dropout']:g} / target {100 * settings['target_dropout']:g}%\n"
            f"Bias XYZ (mm): tool {tuple(map(float, settings['tool_bias']))} / target {tuple(map(float, settings['target_bias']))}\n"
            f"Drift XYZ (mm/s): tool {tuple(map(float, settings['tool_drift']))} / target {tuple(map(float, settings['target_drift']))}\n"
            f"Breathing: {settings['motion_amplitude']:g} mm / {settings['motion_frequency']:g} Hz · "
            f"Proximity weight {settings['proximity_weight']:g} · Uncertainty k {settings['uncertainty_scale']:g}")
        metrics = [
            ("Final true error · all trials (mm)", "actual_error"),
            ("Shaft clearance bound (mm)", "minimum_clearance"),
            ("Travel · successful runs (mm)", "successful_travel"),
            ("Completion · successful runs (s)", "successful_completion_time"),
            ("Planning time (s)", "planning_time"),
            ("Tool raw RMSE (mm)", "tool_raw_rmse"),
            ("Tool filtered RMSE (mm)", "tool_filtered_rmse"),
            ("Tool predicted RMS uncertainty (mm)", "tool_predicted_rms"),
            ("Tool observed 95% coverage (%)", "tool_coverage"),
            ("Target raw RMSE (mm)", "target_raw_rmse"),
            ("Target filtered RMSE (mm)", "target_filtered_rmse"),
            ("Target predicted RMS uncertainty (mm)", "target_predicted_rms"),
            ("Target observed 95% coverage (%)", "target_coverage"),
        ]
        labels = ["Completed trials", "True arrivals", "False arrivals", "Outcomes"]
        self.comparison_table.setRowCount(len(labels) + len(metrics))
        for row, label in enumerate(labels + [label for label, _ in metrics]):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(label))
        for column, summary in enumerate(result["summary"].values(), 1):
            rate = summary["success_rate"]
            values = [str(summary["trials"]),
                      f"{summary['successes']} / {summary['trials']}"
                      + (f" ({100 * rate:.0f}%)" if rate is not None else ""),
                      str(summary["false_arrivals"]),
                      "\n".join(f"{count} × {status.replace('_', ' ')}"
                                for status, count in summary["statuses"].items()) or "—"]
            for _, key in metrics:
                metric = summary["metrics"][key]
                values.append(f"{FormatReading(metric['mean'])} ± {FormatReading(metric['sd'])} "
                              f"(n={metric['count']})")
            for row, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.comparison_table.setItem(row, column, item)
        self.comparison_table.resizeRowsToContents()
        self.comparison_results_button.show()
        self.comparison_dialog.show()

    def plan_path(self):
        if self.comparison_future is not None:
            self.message_label.setText("Stop the comparison before planning a route.")
            return
        source = self.navigation_mode_combo.currentIndex()
        mode = self.planning_mode_combo.currentData()
        if not self.TrackingAvailable(source, mode):
            self.message_label.setText("Required tracking is unavailable. Wait for fresh readings before planning.")
            return
        self.stop_path()
        self.pressed_keys.clear()
        self.plan_source_index = source
        self.plan_source_name = self.navigation_mode_combo.currentText()
        self.plan_mode = mode
        self.plan_mode_name = self.planning_mode_combo.currentText()
        self.plan_uncertainty_scale = self.uncertainty_scale.value()
        tool_offset = None
        tool_covariance = None
        if self.plan_mode == "uncert_aware":
            # compares the tip with nominal geometry from the same sensor sample
            tool_offset = (self.tool_estimate.copy()
                           - tip_position(self.port, *self.sampled_configuration))
            tool_covariance = self.tool_filter.covariance[:3, :3].copy()
        # freezes selected target estimate; new observations don't move this route
        self.planned_target = (self.sampled_target, self.target_measurement,
                               self.target_estimate)[self.plan_source_index].copy()
        self.planning_cancel = Event()
        # searches with copied scene values; only the UI thread updates widgets
        self.planning_future = self.executor.submit(
            find_path, self.configuration.copy(), self.planned_target.copy(), self.port.copy(),
            self.tool_radius, self.structure_centre.copy(), self.structure_radius,
            self.lower_limits.copy(), self.upper_limits.copy(), self.required_clearance,
            self.target_tolerance, cancel_event=self.planning_cancel,
            include_target=True, mode=self.plan_mode, proximity_weight=self.proximity_weight.value(),
            tool_offset=tool_offset, tool_covariance=tool_covariance,
            uncertainty_scale=self.plan_uncertainty_scale)
        self.plan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.plan_label.setText("Searching the configuration grid…")
        self.margin_label.setText("Calculating the frozen tool-tracking margin…")
        self.margin_label.setVisible(self.plan_mode == "uncert_aware")
        self.route_result_label.setText(f"Target snapshot: {self.plan_source_name}.\nNew readings do not move the planned route.")
        self.route_result_label.setVisible(True)
        self.message_label.setText(f"{self.plan_mode_name} with {self.plan_source_name.lower()}. Cancel or use manual controls to interrupt.")

    def poll_planning(self):
        if self.planning_future is None or not self.planning_future.done():
            return
        future = self.planning_future
        self.planning_future = None
        self.planning_cancel = None
        self.plan_button.setEnabled(self.comparison_future is None)
        try:
            result = future.result()
        except Exception as error:
            self.cancel_button.setEnabled(False)
            self.margin_label.hide()
            self.route_result_label.hide()
            self.plan_label.setText("Planning failed.")
            self.message_label.setText(f"Planning failed: {error}")
            return
        self.plan_label.setText(f"Search: {result['time']:.3f} s\n"
                               f"Expanded configurations: {result['expanded']}")
        self.margin_label.setText(f"Frozen margin · Base {result['base_clearance']:.2f} mm\n"
                                  f"Extra {result['extra_clearance']:.2f} mm · Total {result['required_clearance']:.2f} mm")
        if result["status"] != "found":
            messages = {
                "invalid_start": "The starting configuration violates limits or required clearance.",
                "insufficient_clearance": f"The start lacks the selected {result['required_clearance']:.2f} mm clearance. Reduce the uncertainty factor, improve tracking, or choose a different start.",
                "invalid_target": "The target region is inaccessible within the limits or required clearance.",
                "no_grid_goal": "No configuration on this grid meets the target tolerance and clearance.",
                "no_path": "No route connects the start to a valid goal on this grid.",
                "budget_exceeded": "Search budget exhausted; a route may still exist.",
                "cancelled": "Planning cancelled.",
            }
            self.cancel_button.setEnabled(False)
            self.route_result_label.hide()
            self.message_label.setText(messages[result["status"]])
            return
        self.path = result["path"]
        self.route_clearance = result["required_clearance"]
        radius_scale = ((self.structure_radius + self.route_clearance)
                        / (self.structure_radius + self.required_clearance))
        self.margin_actor.SetScale(radius_scale)
        self.render_pending = True
        score = (f"Weighted score: {result['cost']:.2f}\n"
                 if self.plan_mode != "conventional" else "")
        self.plan_label.setText(f"Planned tip travel: {result['length']:.2f} mm\n"
                               f"Clearance bound: ≥ {result['minimum_clearance']:.2f} mm\n"
                               f"{score}Search: {result['time']:.3f} s · Expanded: {result['expanded']}")
        self.plan_label.setToolTip("Travel is the planned tip path length. Clearance is a conservative lower bound across the whole shaft + every movement. The weighted score adds proximity weight × clearance penalty to travel.")
        points = [tip_position(self.port, *self.path[0])]
        for start, end in zip(self.path, self.path[1:]):
            change = end - start
            displacement = abs(change[2]) + max(start[2], end[2]) * sum(abs(change[:2]))
            intervals = max(1, int(np.ceil(displacement / 2.0)))
            for fraction in np.linspace(0, 1, intervals + 1)[1:]:
                points.append(tip_position(self.port, *(start + fraction * change)))
        # samples rotational arcs so the display follows the planned tip movement
        self.route_points = np.array(points)
        mesh = (pv.lines_from_points(self.route_points) if len(points) > 1
                else pv.PolyData(self.route_points))
        self.path_actor = self.viewport.add_mesh(
            mesh, color="#7757b5", line_width=3, render_lines_as_tubes=True,
            point_size=9, render_points_as_spheres=True)
        self.viewport.render()
        self.follow_button.setEnabled(True)
        self.update_route_guidance()
        if len(self.path) == 1:
            self.follow_button.setText("Check arrival")
            self.message_label.setText("No movement needed for this target snapshot. Press Check arrival to verify tracking feedback.")
        else:
            self.message_label.setText("Route ready. Use manual controls with the guide, or Follow route for automatic execution.")

    def follow_path(self):
        if self.path is None:
            return
        if not self.CheckRouteTracking():
            return
        if self.manual_guidance or self.guidance_warning is not None:
            self.message_label.setText("Plan a fresh route for automatic execution. The displayed route remains a reference.")
            return
        if not np.allclose(self.configuration, self.path[0], atol=1e-9, rtol=0):
            self.start_manual_control()
            self.message_label.setText("The instrument moved. Plan a new route from its current position.")
            return
        self.pressed_keys.clear()
        self.pause_button.setChecked(False)
        self.command = self.configuration.copy()
        self.path_index = 1
        self.following = len(self.path) > 1
        self.route_feedback_active = True
        self.route_started_time = perf_counter()
        self.route_finished_time = None if self.following else self.route_started_time
        self.arrival_started = [None, None, None]
        self.arrival_confirmed = [False, False, False]
        self.plan_button.setEnabled(False)
        self.follow_button.setEnabled(False)
        self.message_label.setText("Following the planned route." if self.following else "Checking arrival feedback for 0.5 s.")

    def stop_path(self, _signal=None, *, keep_route=False):
        if self.planning_cancel is not None:
            self.planning_cancel.set()
        if self.planning_future is not None:
            self.planning_future.cancel()
        self.planning_future = None
        self.planning_cancel = None
        self.following = False
        self.route_feedback_active = False
        self.route_finished_time = None
        self.route_result = None
        self.command = self.configuration.copy()
        self.plan_button.setEnabled(self.comparison_future is None)
        self.follow_button.setEnabled(False)
        self.sync_controls()
        if keep_route and self.path is not None:
            self.manual_guidance = True
            self.update_route_guidance()
            self.message_label.setText("Manual control. The route stays visible as a reference; replan for automatic execution.")
            return
        self.route_clearance = self.required_clearance
        if self.margin_actor is not None:
            self.margin_actor.SetScale(1.0)
            self.render_pending = True
        self.planned_target = None
        self.path = None
        self.route_points = None
        self.manual_guidance = False
        self.guidance_warning = None
        self.guidance_warning_pending = None
        self.guidance_warning_started = None
        self.path_index = 0
        if self.path_actor is not None:
            self.viewport.remove_actor(self.path_actor)
            self.path_actor = None
        self.follow_button.setText("Follow route")
        self.cancel_button.setEnabled(False)
        self.plan_label.setText("Grid: 5° yaw / pitch + 5 mm insertion\nTarget tolerance: 2 mm")
        self.plan_label.setToolTip("")
        self.margin_label.clear()
        self.margin_label.hide()
        self.route_result_label.clear()
        self.route_result_label.hide()
        self.message_label.setText("Route cancelled. Ready for manual movement.")

    def start_manual_control(self):
        # stops execution once; later commands keep the other manual axes
        if self.planning_future is not None:
            self.stop_path()
        elif self.path is not None and not self.manual_guidance:
            self.stop_path(keep_route=True)

    def set_command(self, axis):
        value = self.sliders[axis].value() / 10
        self.start_manual_control()
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
        self.scenario_description.setText("Random start within the scenario limits.")
        self.update_instrument()
        self.reset_measurements()
        self.message_label.setText("Instrument reset to a new random starting position. Plan a new route or use manual controls.")

    def reset_measurements(self):
        if self.path is not None or self.planning_future is not None:
            self.stop_path()
            self.message_label.setText("Sensors restarted. Plan a new route with the new readings.")
        # keeps dropout draws separate from Gaussian noise + instrument resets
        seed = self.sensor_seed.value()
        self.sensor_generator = np.random.default_rng(seed)
        self.dropout_generators = [np.random.default_rng([seed, index]) for index in (1, 2)]
        self.sensor_drift = np.zeros((2, 3))
        self.measurement_count = 0
        self.tool_filter = None
        self.target_filter = None
        self.tool_measurement = None
        self.target_measurement = None
        self.tool_estimate = None
        self.target_estimate = None
        self.observations = [None, None]
        self.observation_times = [None, None]
        self.measurement_counts = np.zeros(2, dtype=int)
        self.estimate_counts = np.zeros(2, dtype=int)
        self.measurement_squared_error = np.zeros(2)
        self.estimate_squared_error = np.zeros(2)
        self.last_measurement_time = None
        self.arrival_started = [None, None, None]
        self.arrival_confirmed = [False, False, False]
        self.distance_history = deque(maxlen=300)
        self.distance_started_time = perf_counter()
        self.distance_plot_limit = 10.0
        self.sensor_timer.start(100)
        self.update_measurements()
        self.update_sensor_readouts()
        self.readout_timer.start(500)

    def update_measurements(self):
        sample_time = perf_counter()
        self.UpdateRespiratoryMotion(sample_time)
        self.sampled_target = self.target.copy()
        self.sampled_motion = (self.motion_displacement, self.motion_velocity)
        if self.last_measurement_time is not None and sample_time - self.last_measurement_time > self.tracking_timeout:
            self.CheckRouteTracking()
            # restarts confirmation periods when observations are delayed
            self.arrival_started = [None, None, None]
            self.guidance_warning_pending = None
            self.guidance_warning_started = None
        self.sampled_configuration = self.configuration.copy()
        tool_tip = tip_position(self.port, *self.sampled_configuration)
        self.sampled_clearance = shaft_clearance(
            self.port, tool_tip, self.tool_radius,
            self.structure_centre, self.structure_radius)
        timestep = (max(sample_time - self.last_measurement_time, 1e-6)
                    if self.last_measurement_time is not None else None)
        self.measurement_errors = np.full(2, np.nan)
        self.estimate_errors = np.full(2, np.nan)
        filters = [self.tool_filter, self.target_filter]
        readings = [self.tool_measurement, self.target_measurement]
        estimates = [None, None]
        direction = np.eye(3)[self.sensor_axis.currentIndex()]
        for index, (position, noise, dropout, bias, drift, acceleration) in enumerate((
                (tool_tip, self.tool_noise.value(), self.tool_dropout.value(),
                 self.tool_bias.value(), self.tool_drift.value(), 20.0),
                (self.sampled_target, self.target_noise.value(), self.target_dropout.value(),
                 self.target_bias.value(), self.target_drift.value(), 1.0))):
            # accumulates drift while preserving previous offsets when its rate changes
            drift_rate = direction * drift
            biased_position = ApplyBias(position, direction * bias + self.sensor_drift[index],
                                        drift_rate, timestep or 0.0)
            self.sensor_drift[index] += drift_rate * (timestep or 0.0)
            # draws noise every tick so dropout does not change the Gaussian sequence
            observation = ApplyDropout(measure_position(biased_position, noise, self.sensor_generator),
                                       dropout / 100, self.dropout_generators[index])
            self.observations[index] = observation
            estimator = filters[index]
            if estimator is not None:
                estimator.predict(timestep)
            if observation is not None:
                # initialises each filter once from its first received reading
                if estimator is None:
                    estimator = KalmanFilter(observation, noise, acceleration_std=acceleration)
                else:
                    estimator.update(observation)
                readings[index] = observation
                self.observation_times[index] = sample_time
                self.measurement_errors[index] = np.linalg.norm(observation - position)
                self.measurement_squared_error[index] += self.measurement_errors[index] ** 2
                self.measurement_counts[index] += 1
            filters[index] = estimator
            if estimator is not None:
                estimates[index] = estimator.state[:3].copy()
                self.estimate_errors[index] = np.linalg.norm(estimates[index] - position)
                self.estimate_squared_error[index] += self.estimate_errors[index] ** 2
                self.estimate_counts[index] += 1
        self.tool_filter, self.target_filter = filters
        self.tool_measurement, self.target_measurement = readings
        self.tool_estimate, self.target_estimate = estimates
        self.last_measurement_time = sample_time
        self.measurement_count += 1
        for actor, position in zip((self.tool_measurement_actor, self.target_measurement_actor,
                                     self.tool_estimate_actor, self.target_estimate_actor),
                                    (*self.observations, *estimates)):
            if position is not None:
                actor.SetPosition(*position)
        self.toggle_measurements(None, render=False)
        # compares actual, raw + filtered views of the same task
        self.target_feedbacks = [get_target_feedback(tool_tip, self.sampled_target, self.target_tolerance)]
        for positions, covariances in (
                (self.observations, [self.tool_noise.value() ** 2 * np.eye(3),
                                     self.target_noise.value() ** 2 * np.eye(3)]),
                (estimates, [estimator.covariance[:3, :3] if estimator is not None else None
                             for estimator in filters])):
            self.target_feedbacks.append(
                get_target_feedback(*positions, self.target_tolerance, *covariances)
                if all(position is not None for position in positions)
                else {"distance": np.nan, "arrived": False, "uncertainty": np.nan})
        self.distance_history.append((sample_time - self.distance_started_time,
                                      *(feedback["distance"] for feedback in self.target_feedbacks)))
        for index, feedback in enumerate(self.target_feedbacks):
            if not feedback["arrived"] or not self.TrackingAvailable(index):
                self.arrival_started[index] = None
            elif self.arrival_started[index] is None:
                self.arrival_started[index] = sample_time
            self.arrival_confirmed[index] = (self.arrival_started[index] is not None
                                             and sample_time - self.arrival_started[index] >= self.arrival_dwell - 1e-9)

        self.CheckRouteTracking()
        if self.route_feedback_active:
            reported = self.arrival_confirmed[self.plan_source_index]
            expired = (self.route_finished_time is not None
                       and sample_time - self.route_finished_time >= 3.0)
            if reported or expired:
                # uses only selected-source arrival to stop; truth scores the outcome separately
                actual = self.target_feedbacks[0]
                selected = self.target_feedbacks[self.plan_source_index]
                self.route_result = {"source": self.plan_source_name, "reported": reported,
                                     "method": self.plan_mode_name, "required_clearance": self.route_clearance,
                                     "actual_arrived": actual["arrived"],
                                     "actual_distance": actual["distance"],
                                     "reported_distance": selected["distance"],
                                     "time": sample_time - self.route_started_time}
                self.following = False
                self.route_feedback_active = False
                self.command = self.configuration.copy()
                self.sync_controls()
                self.plan_button.setEnabled(self.comparison_future is None)
                if reported:
                    outcome = "Verified arrival" if actual["arrived"] else "False arrival"
                else:
                    outcome = "Arrival missed" if actual["arrived"] else "Arrival not confirmed"
                self.route_result_label.setText(
                    f"{outcome} · {self.plan_source_name}\nActual distance {actual['distance']:.2f} mm")
                self.message_label.setText(f"{outcome} · actual tip–target distance {actual['distance']:.2f} mm. Plan again to use the latest target estimate.")
        self.update_route_guidance(refresh=False)
        if self.measurements_checkbox.isChecked() or self.estimates_checkbox.isChecked():
            self.render_pending = True

    def update_sensor_readouts(self):
        # refreshes one snapshot while sampling + filtering continue at 10 Hz
        self.motion_label.setText(f"Actual Z offset {self.sampled_motion[0]:.2f} mm\n"
                                  f"Actual Z velocity {self.sampled_motion[1]:.2f} mm/s")
        self.toggle_measurements(None, render=False)
        if self.measurements_checkbox.isChecked():
            self.render_pending = True
        guide_distance = self.update_route_guidance()
        rows = ""
        for index, (name, feedback) in enumerate(zip(("Actual", "Raw", "Filtered"), self.target_feedbacks)):
            available = self.TrackingAvailable(index)
            state = ("Within" if feedback["arrived"] else "Outside") if available else "Unavailable"
            if index == 2 and not available and np.isfinite(feedback["distance"]):
                state = "Prediction"
            value = feedback["distance"] if available or index == 2 else None
            rows += (f'<tr><td>{name}</td><td align="right">{FormatReading(value)}</td>'
                     f'<td align="right">{state}</td></tr>')
        self.feedback_distance_label.setText(
            '<table width="100%" cellspacing="4"><tr><td></td><td align="right">mm</td>'
            f'<td align="right">2 mm region</td></tr>{rows}</table>')
        mode = self.navigation_mode_combo.currentIndex()
        selected = self.target_feedbacks[mode]
        if self.route_result is not None:
            self.feedback_status_label.setText(self.route_result_label.text())
        elif not self.TrackingAvailable(mode):
            self.feedback_status_label.setText("Tracking unavailable. Arrival confirmation requires fresh tool + target readings.")
        elif self.arrival_confirmed[mode]:
            self.feedback_status_label.setText("Arrival indication stable for 0.5 s.\nActual result is evaluated separately.")
        elif selected["arrived"]:
            self.feedback_status_label.setText("Within tolerance; awaiting 0.5 s confirmation.")
        else:
            self.feedback_status_label.setText("Selected guidance is outside the target tolerance.")
        self.relative_uncertainty_label.setText(
            f"Relative-position uncertainty · RMS mm\nRaw {FormatReading(self.target_feedbacks[1]['uncertainty'] if self.TrackingAvailable(1) else None)}  ·  Filtered {FormatReading(self.target_feedbacks[2]['uncertainty'])}")
        # plots every sampled distance - drawing stays at 2 Hz rate
        if self.distance_canvas.isVisible():
            history = np.array(self.distance_history)
            for index, line in enumerate(self.distance_lines):
                line.set_data(history[:, 0], history[:, index + 1])
            latest_time = history[-1, 0]
            self.distance_axes.set_xlim(max(0.0, latest_time - 30.0), max(5.0, latest_time))
            self.distance_plot_limit = max(self.distance_plot_limit, 1.15 * float(np.nanmax(history[:, 1:])))
            self.distance_axes.set_ylim(0, self.distance_plot_limit)
            self.distance_canvas.draw_idle()
        readings = [
            (self.tool_measurement_label, "Tool tip", self.tool_measurement, 0, False),
            (self.target_measurement_label, "Target", self.target_measurement, 1, False),
            (self.tool_estimate_label, "Tool tip", self.tool_estimate, 0, True),
            (self.target_estimate_label, "Target", self.target_estimate, 1, True),
        ]
        headings = "".join(f'<td width="33%" align="right">{axis}</td>' for axis in "XYZ")
        now = perf_counter()
        for label, name, position, index, filtered in readings:
            received = self.observation_times[index]
            fresh = (received is not None and self.observations[index] is not None
                     and now - received <= self.tracking_timeout)
            status = "Waiting for first reading" if received is None else (
                f"Fresh · age {now - received:.1f} s" if fresh else
                f"{'Prediction' if filtered else 'Last reading'} · age {now - received:.1f} s")
            values = "".join(f'<td align="right">{FormatReading(value)}</td>'
                             for value in (position if position is not None else (None,) * 3))
            label.setText(f'<b>{name}</b><br>{status}<table width="100%" cellspacing="4">'
                          f'<tr style="color:#62788c">{headings}</tr>'
                          f'<tr style="font-family:Consolas">{values}</tr></table>')
        self.measurement_error_label.setText(
            f"Raw error · mm\nTool {FormatReading(self.measurement_errors[0])}   Target {FormatReading(self.measurement_errors[1])}")
        self.estimate_error_label.setText(
            f"Filtered error · mm\nTool {FormatReading(self.estimate_errors[0])}   Target {FormatReading(self.estimate_errors[1])}")
        # calculates model RMS uncertainty from the three position variances
        uncertainties = [np.sqrt(max(0.0, np.trace(estimator.covariance[:3, :3])))
                         if estimator is not None else None
                         for estimator in (self.tool_filter, self.target_filter)]
        self.uncertainty_label.setText(
            f"Position uncertainty · RMS mm\nTool {FormatReading(uncertainties[0])}   Target {FormatReading(uncertainties[1])}")
        measured_rmse, filtered_rmse = [
            np.sqrt(np.divide(errors, counts, out=np.full(2, np.nan), where=counts > 0))
            for errors, counts in ((self.measurement_squared_error, self.measurement_counts),
                                   (self.estimate_squared_error, self.estimate_counts))]
        self.rmse_label.setText(
            '<b>Running 3D RMSE · mm</b><table width="100%" cellspacing="4">'
            '<tr><td></td><td align="right">Raw</td><td align="right">Filtered</td></tr>'
            f'<tr><td>Tool</td><td align="right">{FormatReading(measured_rmse[0])}</td>'
            f'<td align="right">{FormatReading(filtered_rmse[0])}</td></tr>'
            f'<tr><td>Target</td><td align="right">{FormatReading(measured_rmse[1])}</td>'
            f'<td align="right">{FormatReading(filtered_rmse[1])}</td></tr></table>'
            f'Raw samples {self.measurement_counts[0]} / {self.measurement_counts[1]}<br>'
            f'Filtered samples {self.estimate_counts[0]} / {self.estimate_counts[1]}')
        self.measurement_count_label.setText(f"Sample {self.measurement_count}\nReceived: tool {self.measurement_counts[0]} / target {self.measurement_counts[1]}")
        yaw, pitch = np.rad2deg(self.sampled_configuration[:2])
        self.position_label.setText(f"ACTUAL INSTRUMENT\nYaw {yaw:.1f}° · Pitch {pitch:.1f}° · Depth {self.sampled_configuration[2]:.1f} mm")
        guide_note = (f"\nTip-to-guide {guide_distance:.2f} mm"
                      + (" · Replan advised" if self.guidance_warning is not None else "")
                      if guide_distance is not None else "")
        distance = selected["distance"] if mode != 1 or self.TrackingAvailable(mode) else None
        prediction_note = ""
        if mode == 2 and not self.TrackingAvailable(mode):
            prediction_note = " · Prediction" if np.isfinite(distance) else " · Unavailable"
        self.target_label.setText(f"{self.navigation_mode_combo.currentText().upper()} DISTANCE\n{FormatReading(distance)} mm{prediction_note}  ·  Actual {self.target_feedbacks[0]['distance']:.2f} mm{guide_note}")
        minimum = self.route_clearance if self.following else self.required_clearance
        guide_margin = (f"\nGuide planned at {self.route_clearance:.2f} mm"
                        if self.path is not None and not self.following else "")
        self.clearance_label.setText(f"SHAFT CLEARANCE\n{self.sampled_clearance:.2f} mm  ·  Movement minimum {minimum:.2f} mm{guide_margin}")

    def TrackingAvailable(self, source, mode="conventional"):
        required = (0, 1) if source != 0 else (0,) if mode == "uncert_aware" else ()
        now = perf_counter()
        return all(self.observations[index] is not None
                   and self.observation_times[index] is not None
                   and now - self.observation_times[index] <= self.tracking_timeout for index in required)

    def CheckRouteTracking(self):
        if self.path is None or self.route_result is not None:
            return True
        if self.TrackingAvailable(self.plan_source_index, self.plan_mode):
            return True
        self.guidance_warning = "Required tracking is unavailable; a fresh plan is needed."
        self.arrival_started = [None, None, None]
        self.arrival_confirmed = [False, False, False]
        if not self.manual_guidance:
            self.stop_path(keep_route=True)
            self.message_label.setText("Automatic execution unavailable: tracking lost. The guide stays visible; plan again after tracking returns.")
        return False

    def update_route_guidance(self, refresh=True):
        if self.route_points is None:
            return
        available = self.CheckRouteTracking()
        target = (self.sampled_target, self.target_measurement,
                  self.target_estimate)[self.plan_source_index]
        reason = None
        if available and np.linalg.norm(target - self.planned_target) > self.target_tolerance:
            reason = "Target tracking differs from the snapshot."
        elif available and self.plan_mode == "uncert_aware":
            offset = self.tool_estimate - tip_position(self.port, *self.sampled_configuration)
            live_margin = get_uncertainty_margin(offset, self.tool_filter.covariance[:3, :3],
                                                 self.plan_uncertainty_scale)
            if round(live_margin, 2) > round(self.route_clearance - self.required_clearance, 2):
                reason = "Tool tracking exceeds the planned margin."
        # keeps brief noise changes from flickering the warning
        if reason != self.guidance_warning_pending:
            self.guidance_warning_pending = reason
            self.guidance_warning_started = self.last_measurement_time if reason is not None else None
        elif (reason is not None and self.guidance_warning is None
              and self.last_measurement_time - self.guidance_warning_started >= 0.5 - 1e-9):
            self.guidance_warning = reason
            self.follow_button.setEnabled(False)
        if not refresh or self.route_result is not None:
            return
        tool = (tip_position(self.port, *self.sampled_configuration),
                self.tool_measurement, self.tool_estimate)[self.plan_source_index]
        if tool is None or (self.plan_source_index == 1 and not available):
            distance = None
        elif len(self.route_points) == 1:
            distance = np.linalg.norm(tool - self.route_points[0])
        else:
            # projects the selected tip estimate to each displayed line segment
            segments = np.diff(self.route_points, axis=0)
            squared_lengths = np.sum(segments ** 2, axis=1)
            fractions = np.zeros(len(segments))
            np.divide(np.sum((tool - self.route_points[:-1]) * segments, axis=1),
                      squared_lengths, out=fractions, where=squared_lengths > 0)
            closest = self.route_points[:-1] + np.clip(fractions, 0, 1)[:, None] * segments
            distance = np.linalg.norm(tool - closest, axis=1).min()
        status = ("Manual reference; replan for automatic execution." if self.manual_guidance
                  else "Automatic execution uses the frozen route." if self.route_feedback_active
                  else "Manual guide or automatic execution available.")
        if self.guidance_warning is not None:
            status = f"Replan advised: {self.guidance_warning}"
        if self.plan_source_index == 2 and not available:
            status += " Tip-to-guide uses prediction."
        if self.manual_guidance and self.sampled_clearance < self.route_clearance:
            status += " Below planned clearance; manual minimum still applies."
        self.route_result_label.setText(
            f"{self.plan_source_name} · Tip-to-guide {FormatReading(distance)} mm\n{status}")
        return float(distance) if distance is not None else None

    def toggle_measurements(self, _visible, *, render=True):
        now = perf_counter()
        for index, actor in enumerate((self.tool_measurement_actor, self.target_measurement_actor)):
            actor.SetVisibility(self.measurements_checkbox.isChecked()
                                and self.observations[index] is not None
                                and now - self.observation_times[index] <= self.tracking_timeout)
        for actor, estimator in ((self.tool_estimate_actor, self.tool_filter),
                                 (self.target_estimate_actor, self.target_filter)):
            actor.SetVisibility(self.estimates_checkbox.isChecked() and estimator is not None)
        if render:
            self.viewport.render()

    def toggle_labels(self, visible):
        self.label_actor.SetVisibility(visible)
        self.viewport.render()

    def advance_movement(self):
        try:
            now = perf_counter()
            self.UpdateRespiratoryMotion(now)
            # limits delayed frames to prevent sudden movement jumps
            timestep = min(now - self.last_tick, 0.05)
            self.last_tick = now
            self.poll_planning()
            if self.route_feedback_active and not self.CheckRouteTracking():
                return
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
            clearance = self.route_clearance if self.following else self.required_clearance
            if movement_is_clear(self.port, self.configuration, proposed, self.tool_radius,
                                 self.structure_centre, self.structure_radius,
                                 clearance):
                self.configuration = proposed
                self.update_instrument()
                if self.following:
                    self.command = self.configuration.copy()
                    self.sync_controls()
                    if fraction == 1.0 or np.array_equal(self.configuration, self.path[self.path_index]):
                        self.path_index += 1
                    if self.path_index == len(self.path):
                        self.following = False
                        self.route_finished_time = perf_counter()
                        self.message_label.setText("Route movement finished. Checking arrival feedback for up to 3 s.")
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
                    if event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                        self.pressed_keys.discard(event.key())
                    return super().eventFilter(watched, event)
                if event.type() == QEvent.Type.ShortcutOverride:
                    event.accept()
                elif not event.isAutoRepeat():
                    if event.type() == QEvent.Type.KeyPress and not self.pause_button.isChecked():
                        self.start_manual_control()
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
        self.comparison_cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
        QApplication.instance().removeEventFilter(self)
        self.viewport.close()
        self.distance_canvas.close()
        super().closeEvent(event)
