from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone
import json
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from threading import Event
from time import perf_counter

import numpy as np
import pyvista as pv
from PIL import Image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
from vtkmodules.vtkFiltersCore import vtkPolyDataTangents

from astar import find_path, get_uncertainty_margin
from capture import CaptureFrame, SaveCapture, SnapshotScene
from experiments import RunStudyWorker
from feedback import GetRouteGuidance, get_target_feedback
from geometry import CheckRemainingRoute, InstrumentRotation, movement_is_clear, shaft_clearance, tip_position
from kalman import KalmanFilter
from motion import GetRespiratoryMotion
from perception import AssessPrediction, LoadInstrument, PredictInstrument, TrainInstrument
from safesight import AssessView, DegradeView
from scenarios import SCENARIOS
from sensors import ApplyBias, ApplyDropout, ApplyOutlier, measure_position


def FormatReading(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}"


def CreateRevolvedMesh(profile, inner_radius=0.0, resolution=64):
    # revolves an axial radius profile + caps either a solid part or an open bore
    profile = np.asarray(profile, dtype=float)
    if (profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2
            or not np.isfinite(profile).all() or np.any(profile[:, 1] <= inner_radius)
            or inner_radius < 0 or resolution < 3):
        raise ValueError("Invalid profile.")
    if inner_radius:
        profile = np.vstack((profile, [[profile[-1, 0], inner_radius],
                                       [profile[0, 0], inner_radius]]))
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    points = np.column_stack((np.repeat(profile[:, 0], resolution),
                              (profile[:, 1, None] * np.cos(angles)).ravel(),
                              (profile[:, 1, None] * np.sin(angles)).ravel()))
    indices = np.arange(len(profile) * resolution).reshape(-1, resolution)
    following = np.roll(indices, -1, axis=0)
    if not inner_radius:
        indices, following = indices[:-1], following[:-1]
    faces = np.stack((np.full_like(indices, 4), indices, np.roll(indices, -1, axis=1),
                      np.roll(following, -1, axis=1), following), axis=-1).ravel()
    if not inner_radius:
        faces = np.r_[faces, resolution, np.arange(resolution)[::-1], resolution,
                       np.arange((len(profile) - 1) * resolution, len(profile) * resolution)]
    return pv.PolyData(points, faces=faces).compute_normals(
        auto_orient_normals=True, consistent_normals=True, split_vertices=True, feature_angle=50).triangulate()


class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Laparoscopic Navigation Simulator")
        self.resize(1360, 860)
        self.setMinimumSize(1000, 650)

        self.port = np.zeros(3)
        # places the wall horizontally + maps insertion down into the body
        self.scene_rotation = np.array([[0.0, 1.0, 0.0],
                                        [0.0, 0.0, -1.0],
                                        [-1.0, 0.0, 0.0]])
        self.scene_matrix = np.eye(4)
        self.scene_matrix[:3, :3] = self.scene_rotation
        self.wall_bounds = (0.0, 2.0, -165.0, 165.0, -185.0, 185.0)
        self.wall_opening_radius = 6.0
        self.scope_port = self.port + np.array([0.0, -70.0, 30.0])
        self.scope_radius = 2.5
        self.scope_depth = 20.0
        self.scope_length = 135.0
        self.scope_opening_radius = 4.5
        self.tool_radius = 1.5
        self.shaft_length = 135.0
        self.jaw_length = 13.0
        self.tip_length = self.jaw_length + 2.0
        self.render_pending = False
        self.capture_folder = None
        self.capture_executor = None
        self.capture_future = None
        self.capture_saving = False
        self.capture_effects = None
        self.sight_result = None
        self.perception_model = None
        self.perception_report = None
        self.perception_report_json = None
        self.perception_result = None
        self.perception_future = None
        self.perception_executor = None
        self.perception_cancel = Event()
        self.calibration_dialog = None
        self.calibration_canvas = None
        self.required_clearance = 2.0
        self.route_clearance = self.required_clearance
        self.execution_clearance = self.required_clearance
        self.route_check_key = None
        self.route_clearance_valid = True
        self.lower_limits = np.array([np.deg2rad(-45), np.deg2rad(-35), 10.0])
        self.upper_limits = np.array([np.deg2rad(45), np.deg2rad(35), 110.0])
        # keeps the scope inside its opening + clear of the grasper's movement envelope
        self.camera_lower_limits = np.array([np.deg2rad(-3), np.deg2rad(-3), self.scope_depth])
        self.camera_upper_limits = np.array([np.deg2rad(3), np.deg2rad(3), 40.0])
        # shares the physical camera's optics between the viewport + saved captures
        self.camera_view_angle = 45.0
        self.camera_clipping_range = (0.5, 1000.0)
        self.speed_limits = np.array([np.deg2rad(20), np.deg2rad(20), 15.0])
        self.camera_speed_limits = self.speed_limits.copy()
        self.camera_speed_limits[:2] = np.deg2rad(3)
        # loads project-relative materials once, including when VS Code uses another working directory
        material_folder = Path(__file__).resolve().parent / "assets" / "materials"
        self.material_textures = {name: pv.read_texture(material_folder / f"{name}.png")
                                  for name in ("skin", "tissue", "liver", "fat")}
        self.anatomy_meshes = {name: pv.read(material_folder.parent / "anatomy" / f"{name}.vtp")
                               for name in ("liver", "gallbladder", "abdomen", "bowel", "diaphragm",
                                            "stomach", "duodenum", "peritoneum", "ducts", "vessels")}
        for texture in self.material_textures.values():
            texture.interpolate = texture.mipmap = texture.repeat = True
            texture.use_srgb_color_space = True
        # adds subtle surface grain through shading, without displacing physical geometry
        grain = np.random.default_rng(41).standard_normal((512, 512))
        for _ in range(4):
            grain = (grain + np.roll(grain, 1, axis=0) + np.roll(grain, -1, axis=0)
                     + np.roll(grain, 1, axis=1) + np.roll(grain, -1, axis=1)) / 5
        grain /= np.max(np.abs(grain))
        dy, dx = np.gradient(grain)
        normals = np.dstack((-dx, -dy, np.ones_like(grain)))
        normals /= np.linalg.norm(normals, axis=2)[:, :, None]
        self.material_normal = pv.Texture(np.rint(127.5 * (normals + 1)).astype(np.uint8))
        roughness = 0.9 + 0.1 * grain
        self.material_roughness = pv.Texture(np.rint(255 * np.dstack(
            (np.ones_like(grain), roughness, np.zeros_like(grain)))).astype(np.uint8))
        for texture in (self.material_normal, self.material_roughness):
            texture.interpolate = texture.mipmap = texture.repeat = True
            texture.use_srgb_color_space = False
        self.target_tolerance = 2.0
        self.arrival_dwell = 0.5
        self.tracking_timeout = 0.25
        self.tracking_paused_time = None
        self.tracking_recovery_samples = 0
        self.tracking_recovery_sample = None
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
        self.study_context = get_context("spawn")
        self.comparison_process = None
        self.comparison_cancel = self.study_context.Event()
        self.comparison_progress = None
        self.comparison_results = None
        self.comparison_response = None
        self.comparison_result = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pressed_keys = set()
        self.key_directions = {
            Qt.Key.Key_A: (0, -1), Qt.Key.Key_D: (0, 1),
            Qt.Key.Key_Q: (1, -1), Qt.Key.Key_E: (1, 1),
            Qt.Key.Key_S: (2, -1), Qt.Key.Key_W: (2, 1),
        }
        self.camera_key_directions = {Qt.Key.Key_C: 1, Qt.Key.Key_V: -1}
        self.camera_mouse_position = None
        self.camera_mouse_sensitivity = np.deg2rad(0.03)

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
        control_layout.setSpacing(6)

        selectors = QFormLayout()
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(list(SCENARIOS))
        selectors.addRow("Scenario", self.scenario_combo)
        self.scenario_description = QLabel(wordWrap=True)
        self.motion_group = QFrame()
        motion_settings = QFormLayout(self.motion_group)
        self.motion_amplitude = QDoubleSpinBox(minimum=0.0, maximum=10.0, decimals=1,
                                              singleStep=0.5, suffix=" mm", keyboardTracking=False)
        self.motion_frequency = QDoubleSpinBox(minimum=0.0, maximum=0.5, decimals=2,
                                              singleStep=0.05, value=0.2, suffix=" Hz",
                                              keyboardTracking=False)
        self.motion_amplitude.setToolTip("Maximum world Y displacement from the resting position. Zero keeps anatomy still.")
        self.motion_frequency.setToolTip("Cycles per second: 0.2 Hz gives one cycle every 5 seconds.")
        self.motion_group.setToolTip(
            "Moves organ + target together along world Y. Changing settings preserves tool and scope poses, "
            "clears the old route and restarts tracking.")
        motion_settings.addRow("Amplitude", self.motion_amplitude)
        motion_settings.addRow("Frequency", self.motion_frequency)
        self.motion_label = QLabel(wordWrap=True)
        motion_settings.addRow(self.motion_label)
        self.navigation_mode_combo = QComboBox()
        self.navigation_mode_combo.addItems(["Ideal positions", "Raw tracking", "Filtered tracking"])
        self.navigation_mode_combo.setCurrentIndex(2)
        self.navigation_mode_combo.setToolTip("Plans towards a target snapshot from this source. Changing source cancels the route.")
        selectors.addRow("Guidance", self.navigation_mode_combo)
        control_layout.addLayout(selectors)

        self.instrument_group = QGroupBox("Instrument commands")
        self.instrument_group.setMinimumHeight(150)
        instrument_layout = QVBoxLayout(self.instrument_group)
        instrument_layout.setSpacing(4)
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
            slider.valueChanged.connect(lambda _, axis=index: self.set_command(axis))
            instrument_layout.addWidget(label)
            instrument_layout.addWidget(slider)
            self.command_labels.append(label)
            self.sliders.append(slider)
        self.keyboard_help = QLabel("W / S  Insert / retract\nA / D  Yaw · Q / E  Pitch", wordWrap=True)
        self.keyboard_help.setStyleSheet("font-size: 9pt; color: #62788c;")
        instrument_layout.addWidget(self.keyboard_help)
        control_layout.addWidget(self.instrument_group)

        planning_group = QGroupBox("A* navigation")
        planning_layout = QVBoxLayout(planning_group)
        planning_layout.setSpacing(6)
        self.planning_mode_combo = QComboBox()
        self.planning_mode_combo.addItem("Conventional A*", "conventional")
        self.planning_mode_combo.addItem("Proximity-aware A*", "prox_aware")
        self.planning_mode_combo.addItem("Uncertainty-aware A*", "uncert_aware")
        self.planning_mode_combo.setToolTip("All methods enforce shaft clearance. Uncertainty mode checks fresh tool margins against the remaining route. Changing method cancels the route.")
        self.proximity_weight = QDoubleSpinBox(minimum=0.0, maximum=10.0, singleStep=0.5,
                                              decimals=1, value=1.0, enabled=False)
        self.proximity_weight.setToolTip("Zero uses travel cost only. Higher weights discourage close passes; the penalty decays over 5 mm above the base clearance.")
        self.uncertainty_scale = QDoubleSpinBox(minimum=0.0, maximum=5.0, singleStep=0.5,
                                               decimals=1, value=1.0, enabled=False)
        self.uncertainty_scale.setToolTip("Adds filtered tool-tip disagreement + k times the largest-direction standard deviation. Zero k keeps the disagreement term. k is an experimental sensitivity, not a confidence percentage.")
        method_row = QFormLayout()
        method_row.addRow("Method", self.planning_mode_combo)
        planning_layout.addLayout(method_row)
        self.plan_button = QPushButton("Plan route", objectName="primary")
        self.plan_button.clicked.connect(self.plan_path)
        self.follow_button = QPushButton("Follow route", enabled=False)
        self.follow_button.setToolTip("Automatically executes a freshly planned route for comparison. Manual controls keep the route as a reference.")
        self.follow_button.clicked.connect(self.follow_path)
        self.cancel_button = QPushButton("Cancel", enabled=False)
        self.cancel_button.clicked.connect(self.stop_path)
        route_buttons = QHBoxLayout()
        route_buttons.addWidget(self.plan_button)
        route_buttons.addWidget(self.follow_button)
        route_buttons.addWidget(self.cancel_button)
        planning_layout.addLayout(route_buttons)
        control_layout.addWidget(planning_group)

        self.planning_options = QFrame()
        details_layout = QVBoxLayout(self.planning_options)
        planning_settings = QFormLayout()
        planning_settings.addRow("Proximity weight", self.proximity_weight)
        planning_settings.addRow("Uncertainty k", self.uncertainty_scale)
        details_layout.addLayout(planning_settings)
        self.plan_label = QLabel("Grid: 5° yaw / pitch + 5 mm insertion\nTarget tolerance: 2 mm",
                                 wordWrap=True, minimumHeight=52)
        details_layout.addWidget(self.plan_label)
        self.margin_label = QLabel(wordWrap=True, visible=False)
        self.margin_label.setToolTip("Shows the margin used for planning. Execution checks fresh tool uncertainty against the remaining route, independently of target guidance.")
        details_layout.addWidget(self.margin_label)
        details_layout.addWidget(self.scenario_description)
        self.route_result_label = QLabel(wordWrap=True, minimumHeight=50, visible=False)
        self.route_result_label.setToolTip("Tip-to-guide distance uses the selected tracking source + the displayed tip polyline. It does not check shaft clearance or prove that the instrument follows the planned configurations. Tracking changes must persist for 0.5 s before a replan warning stays visible.")

        self.pause_button = QPushButton("Pause movement", checkable=True)
        self.pause_button.setToolTip("Pauses movement; anatomy + sensors continue.")
        self.pause_button.toggled.connect(self.set_paused)
        self.reset_button = QPushButton("Reset instrument")
        self.reset_button.setToolTip("Places the instrument at a different random starting position.")
        self.reset_button.clicked.connect(self.reset_instrument)
        movement_buttons = QHBoxLayout()
        movement_buttons.addWidget(self.pause_button)
        movement_buttons.addWidget(self.reset_button)
        control_layout.addLayout(movement_buttons)
        self.motion_section = self.CreateExpandableSection("Respiratory motion", self.motion_group)
        self.planning_section = self.CreateExpandableSection("Planning details", self.planning_options)
        control_layout.addWidget(self.motion_section)
        control_layout.addWidget(self.planning_section)
        control_layout.addStretch()
        scene_note = QLabel("Tool radius: 1.5 mm\nRequired shaft clearance: 2.0 mm", wordWrap=True)
        details_layout.addWidget(scene_note)

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
        self.sensor_seed.setToolTip("Repeats Gaussian noise, dropout + outlier sequences.")
        sensor_settings.addRow("Sensor seed", self.sensor_seed)
        sensor_layout.addWidget(sensor_group)
        bias_group = QGroupBox("Sensor bias + drift")
        bias_settings = QFormLayout(bias_group)
        self.sensor_axis = QComboBox()
        self.sensor_axis.addItems(["X", "Y", "Z"])
        self.sensor_axis.setToolTip("World axis shown in the viewport for fixed bias + subsequent drift increments.")
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
        outlier_group = QGroupBox("Sensor outliers")
        outlier_settings = QFormLayout(outlier_group)
        self.tool_outlier = QDoubleSpinBox(minimum=0.0, maximum=100.0, decimals=0,
                                          singleStep=1, suffix=" %", keyboardTracking=False)
        self.target_outlier = QDoubleSpinBox(minimum=0.0, maximum=100.0, decimals=0,
                                            singleStep=1, suffix=" %", keyboardTracking=False)
        for name, setting in (("Tool outliers", self.tool_outlier),
                              ("Target outliers", self.target_outlier)):
            setting.setToolTip("Probability of adding a spike to each reading. Changes preserve the filter.")
            outlier_settings.addRow(name, setting)
        self.outlier_size = QDoubleSpinBox(minimum=0.0, maximum=50.0, value=20.0,
                                          decimals=1, singleStep=1, suffix=" mm", keyboardTracking=False)
        self.outlier_size.setToolTip("3D spike distance in a random direction, added to ordinary sensor error.")
        outlier_settings.addRow("Spike size", self.outlier_size)
        self.reject_outliers = QCheckBox("Reject outliers", checked=False)
        self.reject_outliers.setToolTip("After the three-reading startup check, rejects readings outside the prediction's nominal 99% region. Valid motion can also be rejected.")
        self.reject_outliers.toggled.connect(self.stop_path)
        outlier_settings.addRow(self.reject_outliers)
        self.rejected_label = QLabel(wordWrap=True)
        outlier_settings.addRow(self.rejected_label)
        sensor_layout.addWidget(outlier_group)
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
        measurements_group = QGroupBox("Raw world positions · mm")
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
        filter_group = QGroupBox("Filtered world positions · mm", visible=False)
        filter_layout = QVBoxLayout(filter_group)
        self.tool_estimate_label = QLabel(wordWrap=True)
        self.target_estimate_label = QLabel(wordWrap=True)
        self.estimate_error_label = QLabel(wordWrap=True)
        self.uncertainty_label = QLabel(wordWrap=True)
        self.rmse_label = QLabel(wordWrap=True)
        self.uncertainty_label.setToolTip("Model RMS position uncertainty: square root of the three position variances added together. Unknown bias, drift + spikes are not included.")
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
        self.sensor_baseline = QLabel(wordWrap=True)
        estimate_layout.addWidget(self.sensor_baseline)
        for setting in (self.tool_noise, self.target_noise, self.sensor_seed):
            setting.valueChanged.connect(self.reset_measurements)

        comparison_layout = self.create_control_tab("Experiments")
        self.comparison_setup_label = QLabel(wordWrap=True)
        comparison_layout.addWidget(self.comparison_setup_label)
        comparison_settings = QFormLayout()
        self.comparison_type = QComboBox()
        self.comparison_type.addItem("Single comparison")
        self.comparison_type.addItem("Tool-noise study", "tool_noise")
        self.comparison_type.addItem("Uncertainty-factor study", "uncertainty_scale")
        self.comparison_type.addItem("Outlier rejection study", "reject_outliers")
        self.comparison_type.currentIndexChanged.connect(self.UpdateComparison)
        comparison_settings.addRow("Experiment", self.comparison_type)
        self.comparison_trials = QSpinBox(minimum=1, maximum=100, value=10,
                                         keyboardTracking=False)
        comparison_settings.addRow("Trials per method", self.comparison_trials)
        comparison_layout.addLayout(comparison_settings)
        self.comparison_button = QPushButton("Run comparison", objectName="primary")
        self.comparison_button.setToolTip("Runs all three methods from the scenario's fixed pose. Studies vary one setting, keeping other settings + seeds fixed. Each trial restarts sensors + breathing, uses a 1 s warmup and a 20 s simulation limit.")
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
        self.comparison_level = QComboBox(visible=False)
        self.comparison_level.currentIndexChanged.connect(self.ShowComparison)
        result_layout.addWidget(self.comparison_level)
        self.comparison_metrics = [
            ("Final true error · all trials (mm)", "actual_error"),
            ("Shaft clearance bound (mm)", "minimum_clearance"),
            ("Travel · successful runs (mm)", "successful_travel"),
            ("Completion · successful runs (s)", "successful_completion_time"),
            ("Planning time (s)", "planning_time"),
            ("Tracking pauses", "tracking_pauses"),
            ("Time paused for tracking (s)", "paused_time"),
            ("Tool raw RMSE (mm)", "tool_raw_rmse"),
            ("Tool filtered RMSE (mm)", "tool_filtered_rmse"),
            ("Tool predicted RMS uncertainty (mm)", "tool_predicted_rms"),
            ("Tool observed 95% coverage (%)", "tool_coverage"),
            ("Target raw RMSE (mm)", "target_raw_rmse"),
            ("Target filtered RMSE (mm)", "target_filtered_rmse"),
            ("Target predicted RMS uncertainty (mm)", "target_predicted_rms"),
            ("Target observed 95% coverage (%)", "target_coverage"),
            ("Tool rejected readings", "tool_rejected"),
            ("Target rejected readings", "target_rejected"),
        ]
        self.comparison_table = QTableWidget(0, 4)
        self.comparison_table.setToolTip(
            "Mean ± sample SD across independent trials; n counts trials.\n"
            "Predicted RMS = sqrt(mean(trace(P))) over the filtered RMSE samples.\n"
            "Coverage counts errors inside the nominal 95% covariance region, including warmup + dropout predictions.\n"
            "Trials can stop early, so coverage describes their observed samples.\n"
            "Rejected counts include any inconsistent reading; they depend on trial duration.")
        self.comparison_table.setHorizontalHeaderLabels(
            ["Metric / mean ± SD (n)", "Conventional A*", "Proximity-aware A*", "Uncertainty-aware A*"])
        self.comparison_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.verticalHeader().hide()
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.comparison_tabs = QTabWidget()
        self.comparison_tabs.tabBar().setAutoHide(True)
        self.comparison_tabs.addTab(self.comparison_table, "Results at selected level")
        self.comparison_tabs.currentChanged.connect(self.ShowComparison)
        result_layout.addWidget(self.comparison_tabs)
        self.sensitivity_graph = QWidget()
        graph_layout = QVBoxLayout(self.sensitivity_graph)
        self.sensitivity_metric = QComboBox()
        for label, key in [("Successful completion (%)", "success_rate"),
                           ("False arrivals (%)", "false_arrival_rate")] + self.comparison_metrics:
            self.sensitivity_metric.addItem(label, key)
        self.sensitivity_metric.currentIndexChanged.connect(self.PlotSensitivity)
        graph_layout.addWidget(self.sensitivity_metric)
        self.sensitivity_canvas = FigureCanvasQTAgg(Figure(facecolor="white", layout="constrained"))
        self.sensitivity_axes = self.sensitivity_canvas.figure.add_subplot(111)
        self.sensitivity_canvas.setToolTip(
            "Points compare complete study levels using the same seeds. Bars show sample SD, not confidence intervals.\n"
            "Rates use completed trials. Missing measurements leave gaps.\n"
            "Uncertainty k affects only uncertainty-aware A*. With ideal tracking, tool noise affects sensor evaluation + the uncertainty margin.")
        graph_layout.addWidget(self.sensitivity_canvas)
        self.comparison_results_button = QPushButton("View results", visible=False)
        self.comparison_results_button.clicked.connect(self.comparison_dialog.show)
        comparison_layout.addWidget(self.comparison_results_button)
        comparison_layout.addStretch()

        capture_layout = self.create_control_tab("SafeSight")
        sight_group = QGroupBox("View quality")
        sight_settings = QVBoxLayout(sight_group)
        self.sight_checkbox = QCheckBox("Monitor Camera view", checked=True)
        self.sight_checkbox.toggled.connect(self.UpdateSafeSight)
        sight_settings.addWidget(self.sight_checkbox)
        self.sight_status_label = QLabel("Select Camera view", wordWrap=True)
        self.sight_status_label.setStyleSheet("font-weight: 600;")
        sight_settings.addWidget(self.sight_status_label)
        self.sight_advice_label = QLabel(wordWrap=True)
        sight_settings.addWidget(self.sight_advice_label)
        self.sight_metrics_label = QLabel(wordWrap=True)
        sight_settings.addWidget(self.sight_metrics_label)
        self.sight_preview = QLabel("Select Camera view", alignment=Qt.AlignmentFlag.AlignCenter, wordWrap=True)
        self.sight_preview.setFixedSize(240, 126)
        self.sight_preview.setStyleSheet("background: #edf2f6; border: 1px solid #d9e1e8; border-radius: 3px;")
        sight_settings.addWidget(self.sight_preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.sight_preview_label = QLabel("Assessed image centre · sampled at 1 Hz", wordWrap=True)
        sight_settings.addWidget(self.sight_preview_label)
        sight_settings.addWidget(QLabel(
            "Rule-based checks of the central 80% width and height. Annotations can affect detail; "
            "anatomy and occlusion are not assessed.", wordWrap=True))
        capture_layout.addWidget(sight_group)
        perception_panel = QWidget()
        perception_layout = QVBoxLayout(perception_panel)
        perception_layout.setContentsMargins(0, 0, 0, 0)
        self.perception_checkbox = QCheckBox("Show instrument candidates", checked=False, enabled=False)
        self.perception_checkbox.toggled.connect(self.UpdateSafeSight)
        perception_layout.addWidget(self.perception_checkbox)
        perception_layout.addWidget(QLabel(
            "Colour-based prototype; new lighting or haze can cause many false positives. "
            "Predictions are withheld outside the recorded appearance range or when quality is poor. "
            "Reference matching does not establish prediction correctness.", wordWrap=True))
        self.perception_status_label = QLabel("No trained model loaded.", wordWrap=True)
        perception_layout.addWidget(self.perception_status_label)
        self.perception_prediction_label = QLabel("Prediction off", wordWrap=True)
        perception_layout.addWidget(self.perception_prediction_label)
        self.calibration_button = QPushButton("View pixel probability calibration", enabled=False)
        self.calibration_button.setToolTip("Compares pixel probabilities with labelled test images; this is not whole-view confidence.")
        self.calibration_button.clicked.connect(self.PlotCalibration)
        perception_layout.addWidget(self.calibration_button)
        self.perception_metrics_label = QLabel(wordWrap=True)
        perception_layout.addWidget(self.perception_metrics_label)
        self.perception_load_button = QPushButton("Load instrument model")
        self.perception_load_button.clicked.connect(self.LoadPerception)
        perception_layout.addWidget(self.perception_load_button)
        self.perception_train_button = QPushButton("Train from capture folder")
        self.perception_train_button.setToolTip(
            "Needs at least eight distinct camera/tool snapshots. Uses calibration/ captures when present, "
            "otherwise reserves separate snapshots. Saves instrument.npz in the capture folder.")
        self.perception_train_button.clicked.connect(self.TrainPerception)
        perception_layout.addWidget(self.perception_train_button)
        self.perception_cancel_button = QPushButton("Cancel training", enabled=False)
        self.perception_cancel_button.clicked.connect(lambda: self.perception_cancel.set())
        perception_layout.addWidget(self.perception_cancel_button)
        capture_layout.addWidget(self.CreateExpandableSection("Instrument perception", perception_panel))
        sight_tests = QWidget()
        sight_test_layout = QVBoxLayout(sight_tests)
        sight_test_layout.setContentsMargins(0, 0, 0, 0)
        self.sight_effects_checkbox = QCheckBox("Apply image test effects", checked=False)
        self.sight_effects_checkbox.toggled.connect(self.UpdateSafeSight)
        sight_test_layout.addWidget(self.sight_effects_checkbox)
        sight_controls = QWidget(enabled=False)
        sight_controls_layout = QFormLayout(sight_controls)
        sight_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.sight_blur = QDoubleSpinBox(minimum=0, maximum=100, singleStep=10, decimals=0, suffix="%")
        self.sight_exposure = QDoubleSpinBox(minimum=-4, maximum=4, singleStep=0.5, decimals=1, suffix=" EV")
        self.sight_haze = QDoubleSpinBox(minimum=0, maximum=100, singleStep=10, decimals=0, suffix="%")
        self.sight_blur.setToolTip("Gaussian blur; strength scales with image width.")
        self.sight_exposure.setToolTip("Each EV doubles or halves linear-light brightness.")
        self.sight_haze.setToolTip("Blends a neutral veil into the image to test loss of contrast.")
        for label, control in (("Blur", self.sight_blur), ("Exposure", self.sight_exposure), ("Haze", self.sight_haze)):
            sight_controls_layout.addRow(label, control)
            control.valueChanged.connect(lambda _: self.UpdateSafeSight(reset=True))
        self.sight_effects_checkbox.toggled.connect(sight_controls.setEnabled)
        sight_test_layout.addWidget(sight_controls)
        sight_test_layout.addWidget(QLabel(
            "Effects apply to the sampled preview. The 3D workspace stays clear.\n"
            "Captures also save a matching degraded image. Haze is a contrast test, not physical smoke.", wordWrap=True))
        capture_layout.addWidget(self.CreateExpandableSection("Camera image tests", sight_tests))
        capture_group = QGroupBox("Camera capture")
        capture_settings = QVBoxLayout(capture_group)
        capture_settings.addWidget(QLabel("Captures the laparoscope's current view from any perspective.", wordWrap=True))
        self.capture_folder_label = QLabel("No folder selected", wordWrap=True)
        capture_settings.addWidget(self.capture_folder_label)
        self.capture_folder_button = QPushButton("Choose capture folder")
        self.capture_folder_button.clicked.connect(self.ChooseCaptureFolder)
        capture_settings.addWidget(self.capture_folder_button)
        self.capture_resolution = QComboBox()
        self.capture_resolution.addItem("640 × 480", (640, 480))
        self.capture_resolution.addItem("1280 × 720", (1280, 720))
        capture_settings.addWidget(self.capture_resolution)
        self.capture_button = QPushButton("Capture frame", objectName="primary", enabled=False)
        self.capture_button.clicked.connect(self.CaptureCameraFrame)
        capture_settings.addWidget(self.capture_button)
        self.capture_status_label = QLabel(
            "Saves a clean image + ground-truth labels. Enable image tests to also save a degraded copy.", wordWrap=True)
        capture_settings.addWidget(self.capture_status_label)
        capture_layout.addWidget(capture_group)
        capture_layout.addStretch()
        self.capture_timer = QTimer(self)
        self.capture_timer.setSingleShot(True)
        self.capture_timer.timeout.connect(self.CaptureCameraFrame)

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
        self.view_combo.setToolTip("Camera view: left-drag to aim the scope; C / V advance / retract. Tool keys + sliders stay active.")
        self.view_combo.addItems(["Perspective", "Camera view", "Top", "Front", "Overview"])
        self.view_combo.currentIndexChanged.connect(self.reset_camera)
        scene_toolbar.addWidget(self.view_combo)
        view_button = QPushButton("Reset view")
        view_button.clicked.connect(lambda: self.reset_camera())
        scene_toolbar.addWidget(view_button)
        scene_layout.addLayout(scene_toolbar)
        self.guidance_card = QFrame()
        self.guidance_card.setStyleSheet("QFrame { background: white; border-bottom: 1px solid #d9e1e8; }")
        guidance_layout = QVBoxLayout(self.guidance_card)
        guidance_layout.setContentsMargins(12, 8, 12, 8)
        guidance_layout.setSpacing(4)
        guidance_header = QHBoxLayout()
        guidance_title = QLabel("Guidance")
        guidance_title.setStyleSheet("font-size: 11pt; font-weight: 600; border: none;")
        guidance_header.addWidget(guidance_title)
        guidance_header.addStretch()
        self.guidance_status_label = QLabel()
        guidance_header.addWidget(self.guidance_status_label)
        guidance_layout.addLayout(guidance_header)
        self.route_result_label.setStyleSheet("border: none;")
        guidance_layout.addWidget(self.route_result_label)
        self.message_label = QLabel("Ready for manual movement.", wordWrap=True)
        self.message_label.setStyleSheet("border: none;")
        guidance_layout.addWidget(self.message_label)
        self.sight_indicator = QLabel(wordWrap=True, visible=False)
        self.sight_indicator.setToolTip("SafeSight: rule-based exposure and detail checks. See the SafeSight tab for measurements.")
        guidance_layout.addWidget(self.sight_indicator)
        scene_layout.addWidget(self.guidance_card)
        self.viewport = QtInteractor(scene_frame, auto_update=False)
        scene_layout.addWidget(self.viewport.interactor, 1)
        body.addWidget(scene_frame, 1)
        self.viewport.set_background("#edf2f6", top="#fafcfd")
        self.viewport.enable_anti_aliasing("ssaa")
        scene_key = QLabel('<span style="color:#aebfca">●</span> Tool tip &nbsp; '
                           '<span style="color:#9eadb7">━</span> Laparoscope &nbsp; '
                           '<span style="color:#a9394c">●</span> Target &nbsp; '
                           '<span style="color:#a33245">━</span> Blood vessel &nbsp; '
                           '<span style="color:#dbad98">●</span> Body wall &nbsp; '
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
        self.sensor_timer = QTimer(self)
        self.sensor_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.sensor_timer.timeout.connect(self.update_measurements)
        self.readout_timer = QTimer(self)
        self.readout_timer.timeout.connect(self.update_sensor_readouts)
        self.readout_timer.timeout.connect(self.UpdateComparison)
        self.sight_timer = QTimer(self)
        self.sight_timer.setInterval(1000)
        self.sight_timer.timeout.connect(self.UpdateSafeSight)
        self.load_scenario(self.scenario_combo.currentIndex())
        self.scenario_combo.currentIndexChanged.connect(self.load_scenario)
        for setting in (self.motion_amplitude, self.motion_frequency):
            setting.valueChanged.connect(lambda _: self.UpdateRespiratoryMotion(reset=True))
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
        self.sight_timer.start()
        model_path = Path(__file__).resolve().parent / "assets" / "models" / "instrument.npz"
        if model_path.exists():
            self.LoadPerception(model_path)
        # refits once Qt has assigned the viewport dimensions
        QTimer.singleShot(0, lambda: self.reset_camera(self.view_combo.currentIndex()))

    def create_control_tab(self, title):
        panel = QFrame(objectName="controls")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        self.control_tabs.addTab(scroll, title)
        return layout

    def UpdateSafeSight(self, checked=None, *, reset=False):
        # reads the displayed RGB image without changing the scene or rendering again
        self.UpdatePerception()
        self.sight_result = None
        self.sight_preview.clear()
        metrics = ""
        colour = "#62788c"
        active = self.sight_checkbox.isChecked() and self.view_combo.currentText() == "Camera view"
        if not self.sight_checkbox.isChecked():
            title, advice = "Monitoring off", "Enable monitoring to check the Camera image."
        elif not active:
            title, advice = "Select Camera view", "The monitor checks the laparoscope image in Camera view."
        elif reset or checked is not None:
            title, advice = "Waiting for image", "A new image check will appear within a second."
        elif not self.isVisible() or self.isMinimized() or not self.viewport.isVisible():
            title, advice = "Image unavailable", "Restore the Camera view to resume image checks."
        else:
            try:
                image = np.asarray(self.viewport.image)
                if self.sight_effects_checkbox.isChecked():
                    # bounds preview processing; saved pairs retain their selected resolution
                    height, width = image.shape[:2]
                    if width > 640:
                        image = np.asarray(Image.fromarray(image[:, :, :3]).resize(
                            (640, max(16, round(height * 640 / width))), Image.Resampling.LANCZOS))
                    image = DegradeView(image[:, :, :3], blur=self.sight_blur.value() / 100,
                                        exposure=self.sight_exposure.value(), haze=self.sight_haze.value() / 100)
                height, width = image.shape[:2]
                view_image = image[:, :, :3]
                # avoids most corner decorations while retaining the main camera image
                image = image[height // 10:height - height // 10,
                              width // 10:width - width // 10, :3].copy()
                result = AssessView(image)
                self.sight_result = result
                annotations = any(actor is not None and actor.GetVisibility() for actor in (
                    self.label_actor, self.path_actor, self.tool_measurement_actor,
                    self.target_measurement_actor, self.tool_estimate_actor, self.target_estimate_actor))
                overlay = self.UpdatePerception(image, annotated=annotations, view_image=view_image, quality=result)
                preview = QImage(overlay.data, overlay.shape[1], overlay.shape[0], overlay.strides[0],
                                 QImage.Format.Format_RGB888)
                self.sight_preview.setPixmap(QPixmap.fromImage(preview).scaled(
                    self.sight_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            except Exception:
                self.sight_result = None
                title, advice = "Image unavailable", "Waiting for a readable Camera image."
            else:
                self.sight_result = result
                messages = {"dark": "Image is dark; inspect lighting and framing.",
                            "bright": "Image is washed out; check bright regions and framing.",
                            "low_detail": "Low image detail; inspect focus and framing."}
                if result["issues"]:
                    title = "Check camera image"
                    advice = " ".join(messages[issue] for issue in result["issues"])
                    colour = "#97580c"
                elif annotations:
                    title, advice = "Annotated camera image", "Visible labels or guidance can affect the detail check."
                    colour = "#97580c"
                else:
                    title, advice = "No exposure/detail warning", "These checks do not establish visibility of the tool or anatomy."
                if annotations and result["issues"]:
                    advice += " Visible annotations can affect detail."
                metrics = (f"Mean brightness: {result['mean_luminance']:.0%}\n"
                           f"Dark pixels: {result['dark_fraction']:.0%} · Bright pixels: {result['bright_fraction']:.0%}\n"
                           f"Local detail: {result['detail']:.4f}")
        self.sight_status_label.setText(title)
        self.sight_status_label.setStyleSheet(f"font-weight: 600; color: {colour};")
        self.sight_advice_label.setText(advice)
        self.sight_metrics_label.setText(metrics)
        self.sight_indicator.setVisible(active)
        test_image = self.sight_effects_checkbox.isChecked()
        source = "Test image centre" if test_image else "Assessed image centre"
        if self.perception_result is not None:
            source += " + candidates" if self.perception_result["status"] == "shown" else " · prediction withheld"
        self.sight_preview_label.setText(f"{source} · sampled at 1 Hz")
        if self.sight_result is None:
            self.sight_preview.setText(title)
            self.UpdatePerception()
        self.sight_indicator.setText(f"SafeSight · {'Image test · ' if test_image else ''}{title}")
        if self.perception_result is not None and self.perception_result["status"] == "withheld":
            self.sight_indicator.setText("SafeSight · Prediction withheld · " + self.perception_result["message"])
            colour = "#97580c"
        self.sight_indicator.setStyleSheet(f"border: none; font-size: 9pt; color: {colour};")

    def TrainPerception(self):
        if self.perception_future is not None:
            return
        if self.capture_future is not None:
            self.perception_status_label.setText("Finish the camera capture before training.")
            return
        if self.capture_folder is None:
            self.ChooseCaptureFolder()
        if self.capture_folder is None:
            return
        if self.perception_executor is None:
            self.perception_executor = ThreadPoolExecutor(max_workers=1)
        self.perception_cancel.clear()
        output = self.capture_folder / "instrument.npz"
        self.perception_future = self.perception_executor.submit(
            TrainInstrument, self.capture_folder, output=output, cancel=self.perception_cancel)
        self.perception_status_label.setText("Training from the capture folder; complete snapshots are held out for testing…")
        self.perception_train_button.setEnabled(False)
        self.perception_load_button.setEnabled(False)
        self.perception_cancel_button.setEnabled(True)

    def LoadPerception(self, path=None):
        if self.perception_future is not None:
            return
        if path is None or isinstance(path, bool):
            path, _ = QFileDialog.getOpenFileName(self, "Load instrument model", "", "Instrument model (*.npz)")
        if not path:
            return
        self.perception_model = None
        self.perception_report = None
        self.perception_report_json = None
        self.perception_metrics_label.clear()
        if self.calibration_dialog is not None:
            self.calibration_dialog.close()
        try:
            self.perception_model = LoadInstrument(path)
        except Exception as error:
            self.perception_status_label.setText(f"Model unavailable: {error}")
        else:
            self.perception_status_label.setText(f"Loaded experimental pixel baseline: {Path(path).name}")
        self.perception_checkbox.setEnabled(self.perception_model is not None)
        self.calibration_button.setEnabled(self.perception_model is not None and int(self.perception_model["version"]) == 3)
        self.UpdateSafeSight(reset=True)

    def PlotCalibration(self):
        if self.perception_model is None or int(self.perception_model["version"]) != 3:
            return
        report = (self.perception_report if self.perception_report_json is self.perception_model["report_json"]
                  else json.loads(str(self.perception_model["report_json"].item())))
        calibration = report["calibration"]
        if self.calibration_dialog is None:
            self.calibration_dialog = QDialog(self)
            self.calibration_dialog.setWindowTitle("Pixel probability calibration")
            self.calibration_dialog.resize(940, 760)
            calibration_layout = QVBoxLayout(self.calibration_dialog)
            self.calibration_heading = QLabel(wordWrap=True)
            calibration_layout.addWidget(self.calibration_heading)
            self.calibration_canvas = FigureCanvasQTAgg(Figure(facecolor="white", layout="constrained"))
            calibration_layout.addWidget(self.calibration_canvas, 1)
            self.calibration_table = QTableWidget(4, 7)
            self.calibration_table.setHorizontalHeaderLabels(
                ["Images", "Scores", "Brier ↓", "Log loss ↓", "ECE ↓", "Tool Brier ↓", "Other Brier ↓"])
            for column, tooltip in enumerate((
                    "Held-out image variant", "Original or calibrated probabilities",
                    "Mean squared probability error across all pixels; lower is better.",
                    "Penalises confident mistakes; lower is better.",
                    "Weighted gap between predicted and observed tool fraction in ten bins; lower is better.",
                    "Mean squared error on tool pixels only; prevents the background majority hiding misses.",
                    "Mean squared error on all other pixels.")):
                self.calibration_table.horizontalHeaderItem(column).setToolTip(tooltip)
            self.calibration_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.calibration_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.calibration_table.verticalHeader().setVisible(False)
            calibration_layout.addWidget(self.calibration_table)
            calibration_layout.addWidget(QLabel(
                "All held-out test images are included, even when predictions are withheld. "
                "Pixel counts are correlated within snapshots; these plots do not measure whole-view safety.", wordWrap=True))
        self.calibration_heading.setText(
            f"Fitted on {calibration['groups']} separate calibration snapshots; "
            f"tested on {report['validation_groups']} other snapshots. "
            "Curves compare predicted tool-pixel probability with the observed tool-pixel fraction.")
        self.calibration_canvas.figure.clear()
        axes = self.calibration_canvas.figure.subplots(2, 2)
        row = 0
        for column, variant in enumerate(("clean", "degraded")):
            curve_axis, count_axis = axes[:, column]
            curve_axis.plot([0, 1], [0, 1], color="#9babb7", linestyle="--", linewidth=1)
            for index, (key, label, colour) in enumerate((
                    ("raw", "Original", "#286e9f"), ("calibrated", "Calibrated", "#bc681d"))):
                metrics = calibration[variant][key]
                if metrics is None:
                    for table_column, value in enumerate([variant.capitalize(), label] + ["—"] * 5):
                        self.calibration_table.setItem(row, table_column, QTableWidgetItem(value))
                    row += 1
                    continue
                mean_probability = np.array([np.nan if value is None else value
                                             for value in metrics["bin_mean_probability"]])
                observed = np.array([np.nan if value is None else value
                                     for value in metrics["bin_positive_fraction"]])
                curve_axis.plot(mean_probability, observed, "o-", color=colour, label=label, markersize=4)
                edges = np.asarray(metrics["bin_edges"])
                centres = (edges[:-1] + edges[1:]) / 2
                width = (edges[1] - edges[0]) * 0.35
                positions = centres + (index - 0.5) * width
                count_axis.bar(positions, metrics["bin_counts"], width=width, color=colour, alpha=0.65, label=label)
                count_axis.scatter(positions, metrics["bin_positive_counts"], color=colour, marker="_", s=75)
                values = [variant.capitalize(), label] + [
                    "—" if metrics[name] is None else f"{metrics[name]:.5g}"
                    for name in ("brier", "nll", "ece", "positive_brier", "negative_brier")]
                for table_column, value in enumerate(values):
                    self.calibration_table.setItem(row, table_column, QTableWidgetItem(value))
                row += 1
            curve_axis.set(title=variant.capitalize() + " test images", xlim=(0, 1), ylim=(0, 1),
                           xlabel="Predicted P(tool pixel)", ylabel="Observed tool fraction")
            if calibration[variant]["raw"] is None:
                for axis in (curve_axis, count_axis):
                    axis.text(0.5, 0.5, f"No held-out {variant} captures", ha="center", va="center",
                              transform=axis.transAxes)
            else:
                curve_axis.legend(fontsize=8)
            curve_axis.grid(alpha=0.2)
            count_axis.set(xlim=(0, 1), xlabel="Probability bin", ylabel="Pixel count",
                           title="Bars: all pixels · ticks: tool pixels")
            count_axis.set_yscale("symlog", linthresh=1)
            count_axis.grid(axis="y", alpha=0.2)
        self.calibration_dialog.show()
        self.calibration_table.setFixedHeight(
            self.calibration_table.horizontalHeader().height()
            + self.calibration_table.verticalHeader().length() + 2 * self.calibration_table.frameWidth())
        self.calibration_canvas.draw_idle()
        self.calibration_dialog.raise_()

    def UpdatePerception(self, image=None, annotated=False, view_image=None, quality=None):
        # polls CPU training separately from route planning and capture saving
        if self.perception_future is not None and self.perception_future.done():
            try:
                model, _ = self.perception_future.result()
            except Exception as error:
                self.perception_status_label.setText(
                    ("Training cancelled." if self.perception_cancel.is_set() else f"Training failed: {error}")
                    + (" Previous model retained." if self.perception_model is not None else ""))
            else:
                self.perception_model = model
                if self.calibration_dialog is not None:
                    self.calibration_dialog.close()
                self.perception_status_label.setText("Trained experimental pixel baseline; saved instrument.npz in the capture folder.")
                self.perception_checkbox.setEnabled(True)
            self.perception_future = None
            self.perception_train_button.setEnabled(True)
            self.perception_load_button.setEnabled(True)
            self.perception_cancel_button.setEnabled(False)
            self.calibration_button.setEnabled(self.perception_model is not None and int(self.perception_model["version"]) == 3)
        if (self.perception_model is not None
                and self.perception_report_json is not self.perception_model["report_json"]):
            report = json.loads(str(self.perception_model["report_json"].item()))
            lines = [f"Held-out simulator snapshots: {report['validation_groups']}"]
            for variant in ("clean", "degraded"):
                metrics = report.get(variant)
                if metrics is not None:
                    values = ["—" if metrics[key] is None else f"{metrics[key]:.1%}"
                              for key in ("iou", "precision", "recall")]
                    lines.append(f"{variant.capitalize()} captures: IoU {values[0]} · precision {values[1]} · recall {values[2]}")
            negative_rate = report.get("negative_false_positive_fraction")
            if negative_rate is not None:
                lines.append(f"False positive pixels without a tool: {negative_rate:.2%}")
            support = report.get("support") if int(self.perception_model["version"]) >= 2 else None
            if support is not None:
                for variant in ("clean", "degraded"):
                    result = support[variant]
                    accepted_iou = "not assessed" if result["iou"] is None else f"{result['iou']:.1%}"
                    lines.append(f"{variant.capitalize()} shown: {result['accepted']}/{result['frames']} · "
                                 f"shown-only IoU {accepted_iou}")
            self.perception_metrics_label.setText("\n".join(lines))
            self.perception_report = report
            self.perception_report_json = self.perception_model["report_json"]
        self.perception_result = None
        if not self.perception_checkbox.isChecked():
            self.perception_prediction_label.setText("Prediction off")
        elif self.perception_model is None:
            self.perception_prediction_label.setText("No model available")
        elif image is None:
            self.perception_prediction_label.setText("Waiting for a Camera image")
        else:
            try:
                height, width = image.shape[:2]
                # checks full RGB before the preview crop, matching the recorded reference
                gate = AssessPrediction(image if view_image is None else view_image, self.perception_model,
                                        quality=quality)
                reasons = (*gate["reasons"], *(("annotations",) if annotated else ()))
                messages = {"dark": "dark image", "bright": "washed-out image", "low_detail": "low image detail",
                            "appearance_shift": "appearance differs from training examples",
                            "profile_missing": "model has no appearance reference; retrain it",
                            "annotations": "visible annotations"}
                appearance = ("" if gate["distance"] is None else
                              f"\nAppearance difference: {gate['distance']:.3f} · limit {gate['limit']:.3f}")
                if reasons:
                    message = "; ".join(messages[reason] for reason in reasons)
                    self.perception_prediction_label.setText("Prediction withheld: " + message + appearance)
                    self.perception_result = {"status": "withheld", "reasons": reasons, "message": message,
                                              "distance": gate["distance"], "limit": gate["limit"]}
                    return image
                sample = image
                if max(height, width) > 320:
                    scale = 320 / max(height, width)
                    sample = np.asarray(Image.fromarray(image).resize(
                        (max(16, round(width * scale)), max(16, round(height * scale))), Image.Resampling.LANCZOS))
                scores = PredictInstrument(sample, self.perception_model)
                mask = scores >= float(self.perception_model["threshold"])
                fraction = float(mask.mean())
                self.perception_prediction_label.setText(
                    f"Candidate pixels: {fraction:.1%} · experimental"
                    + appearance + "\nMatching appearance does not establish correctness.")
                mask = np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))
                overlay = image.copy()
                overlay[mask] = np.rint(0.5 * overlay[mask] + 0.5 * np.array([50, 215, 235])).astype(np.uint8)
                self.perception_result = {"status": "shown", "candidate_fraction": fraction,
                                          "distance": gate["distance"], "limit": gate["limit"]}
                return overlay
            except Exception as error:
                self.perception_prediction_label.setText(f"Prediction withheld: {error}")
                self.perception_result = {"status": "withheld", "reasons": ("prediction_error",),
                                          "message": "prediction unavailable", "distance": None, "limit": None}
        return image

    def ChooseCaptureFolder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose capture folder", str(self.capture_folder or Path.home()))
        if folder:
            self.capture_folder = Path(folder)
            self.capture_folder_label.setText(self.capture_folder.name or str(self.capture_folder))
            self.capture_folder_label.setToolTip(str(self.capture_folder))
            self.capture_button.setEnabled(True)
            self.capture_status_label.setText("Ready to capture the current laparoscope view.")

    def CaptureCameraFrame(self):
        if self.capture_folder is None:
            return
        try:
            if self.capture_future is None:
                # aligns the visual meshes with the target before freezing their shared snapshot
                self.UpdateRespiratoryMotion()
                # freezes image effects alongside the physical scene snapshot
                self.capture_effects = ({"blur": self.sight_blur.value() / 100,
                                         "exposure": self.sight_exposure.value(),
                                         "haze": self.sight_haze.value() / 100}
                                        if self.sight_effects_checkbox.isChecked() else None)
                # freezes geometry on the UI thread + isolates capture's graphics context
                rotation = np.array(self.scope_actor.user_matrix)[:3, :3]
                camera = pv.Camera()
                camera.position = self.scope_tip.copy()
                camera.focal_point = self.scope_tip + rotation[:, 0]
                camera.up = rotation[:, 2]
                camera.view_angle = self.camera_view_angle
                camera.clipping_range = self.camera_clipping_range
                actors = {
                    "instrument": [self.shaft_actor, self.external_shaft_actor, self.tip_actor,
                                   self.handle_actor, self.collar_actor, self.port_actor],
                    "gallbladder": [self.organ_actor], "liver": [self.posterior_actor],
                    "protected_vessel": [self.vessel_actor], "connective_tissue": [self.tissue_actor],
                    "internal_tissue": [self.cavity_actor, self.bowel_actor, self.peritoneum_actor,
                                        self.vascular_actor],
                    "skin": [self.body_actor, self.wall_actor],
                    "laparoscope": [self.scope_actor, self.scope_port_actor], "bile_duct": [self.duct_actor],
                    "diaphragm": [self.diaphragm_actor], "stomach": [self.stomach_actor],
                    "duodenum": [self.duodenum_actor]}
                metadata = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "scenario": self.scenario_name,
                    "scene_to_world": self.scene_matrix.copy(), "world_units": "mm",
                    "configuration_units": "yaw radians, pitch radians, insertion mm",
                    "camera_configuration": self.camera_configuration.copy(),
                    "tool_configuration": self.configuration.copy(),
                    "tool_tip_world_mm": self.scene_rotation @ tip_position(self.port, *self.configuration),
                    "target_world_mm": self.scene_rotation @ self.target,
                    "vessel_points_world_mm": self.structure_points @ self.scene_rotation.T,
                    "vessel_radius_mm": self.structure_radius,
                    "breathing_amplitude_mm": self.motion_amplitude.value(),
                    "breathing_frequency_hz": self.motion_frequency.value(),
                    "breathing_displacement_mm": self.motion_displacement,
                    "breathing_velocity_mm_s": self.motion_velocity,
                    "target_highlight_removed": True}
                # captures the complete enclosure even when the external inspection window is shown
                enclosure = (self.body_actor, self.cavity_actor, self.wall_actor)
                visibility = [actor.GetVisibility() for actor in enclosure]
                try:
                    for actor in enclosure:
                        actor.SetVisibility(True)
                    actors, camera, light = SnapshotScene(actors, camera, self.scope_light, serializable=True)
                finally:
                    for actor, visible in zip(enclosure, visibility):
                        actor.SetVisibility(visible)
                if self.capture_executor is None:
                    self.capture_executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
                self.capture_saving = False
                self.capture_future = self.capture_executor.submit(
                    CaptureFrame, actors, camera, light, metadata, self.capture_resolution.currentData(),
                    colour_overrides={"gallbladder": np.array([130, 215, 120]) / 255})
                self.capture_status_label.setText("Capturing the laparoscope view…")
            elif self.capture_future.done():
                result = self.capture_future.result()
                if self.capture_saving:
                    image_path, label_path = result
                    self.capture_status_label.setText(
                        f"Saved images/{image_path.name}\n+ labels/{label_path.name}"
                        + (f"\n+ degraded/{image_path.name}" if self.capture_effects is not None else ""))
                    self.capture_future = None
                else:
                    self.capture_future = self.executor.submit(
                        SaveCapture, self.capture_folder, *result, effects=self.capture_effects)
                    self.capture_saving = True
            if self.capture_future is not None:
                self.capture_timer.start(50)
        except Exception as error:
            self.capture_future = None
            if self.capture_executor is not None:
                self.capture_executor.shutdown(wait=False, cancel_futures=True)
                self.capture_executor = None
            self.capture_status_label.setText(f"Capture failed: {error}")
        finally:
            ready = self.capture_future is None
            self.capture_button.setEnabled(ready)
            self.capture_folder_button.setEnabled(ready)
            self.capture_resolution.setEnabled(ready)

    def CreateExpandableSection(self, title, content):
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toggle = QPushButton(f"▸ {title}", checkable=True, objectName="section_toggle")
        toggle.setStyleSheet("text-align: left; padding: 6px;")
        layout.addWidget(toggle)
        layout.addWidget(content)
        content.hide()
        toggle.toggled.connect(content.setVisible)
        toggle.toggled.connect(lambda expanded: toggle.setText(f"{'▾' if expanded else '▸'} {title}"))
        return section

    def UpdateGuidanceCard(self, state=None):
        if self.route_result is not None:
            if self.route_result["reported"]:
                state = "Verified arrival" if self.route_result["actual_arrived"] else "False arrival"
            else:
                state = "Arrival unconfirmed"
        elif self.planning_future is not None:
            state = "Planning"
        elif self.tracking_paused_time is not None:
            state = "Tracking paused"
        elif self.guidance_warning is not None:
            state = "Replan required"
        elif self.route_feedback_active and not self.route_clearance_valid:
            state = "Clearance paused"
        elif self.pause_button.isChecked():
            state = "Paused"
        elif self.path is not None and not self.TrackingAvailable(
                self.plan_source_index, self.plan_mode, accepted=True):
            state = "Tracking unavailable"
        elif self.guidance_warning_pending is not None:
            state = "Tracking changed"
        elif state is None:
            if self.route_feedback_active:
                state = "Following" if self.following else "Checking arrival"
            else:
                state = "Guiding" if self.path is not None else "Manual control"
            if self.message_label.text().startswith(("Movement blocked", "Route stopped:")):
                state = "Blocked"
        colour = "#286e9f"
        if state == "Verified arrival":
            colour = "#177b61"
        elif state in ("False arrival", "Replan required", "Blocked"):
            colour = "#a73842"
        elif state in ("Paused", "Clearance paused", "Tracking paused", "Tracking unavailable",
                       "Tracking changed", "Arrival unconfirmed"):
            colour = "#956414"
        self.guidance_status_label.setText(state)
        self.guidance_status_label.setStyleSheet(
            f"color: {colour}; background: #f3f6f9; border: none; border-radius: 4px;"
            "padding: 4px 8px; font-weight: 600;")
        show_route = (self.path is not None and self.planning_future is None
                      and self.tracking_paused_time is None
                      and (self.route_result is not None or self.guidance_warning is not None
                           or not self.pause_button.isChecked())
                      and not (state == "Blocked" and self.message_label.text().startswith(
                          ("Movement blocked", "Route stopped:"))))
        self.route_result_label.setVisible(show_route)
        self.message_label.setVisible(not show_route)

    def load_scenario(self, index):
        self.scenario_name = self.scenario_combo.itemText(index)
        scenario = SCENARIOS[self.scenario_name]
        self.configuration = scenario["start"].copy()
        self.rest_target = tip_position(self.port, *scenario["target_configuration"])
        self.target = self.rest_target.copy()
        # sets the laparoscope's resting direction at its separate fixed port
        self.scope_direction = self.rest_target + [8.0, 0.0, 0.0] - self.scope_port
        self.scope_direction /= np.linalg.norm(self.scope_direction)
        right = np.cross(self.scope_direction, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        self.scope_matrix = np.eye(4)
        self.scope_matrix[:3, :3] = np.column_stack((self.scope_direction, -right,
                                                  np.cross(right, self.scope_direction)))
        self.scope_matrix[:3, 3] = self.scope_port
        self.camera_configuration = np.array([0.0, 0.0, self.scope_depth])
        self.camera_command = self.camera_configuration.copy()
        self.scope_tip = self.scene_rotation @ (
            self.scope_port + self.scope_depth * self.scope_direction)
        self.motion_started_time = None
        self.structure_points = scenario["structure_points"].copy()
        self.structure_radius = scenario["structure_radius"]
        self.stop_path()
        self.pressed_keys.clear()
        self.pause_button.setChecked(False)
        self.scenario_description.setText(scenario["description"])
        self.viewport.clear()
        self.build_scene()
        self.update_instrument()
        self.reset_camera()
        self.reset_measurements()
        self.message_label.setText(f"{self.scenario_name} loaded. Plan a route or use manual controls.")
        self.UpdateGuidanceCard()

    def build_scene(self):
        self.viewport.enable_lightkit()
        self.viewport.add_light(pv.Light(light_type="headlight", intensity=0.3))
        # lights the operative field while retaining distance + angular falloff
        self.scope_light = pv.Light(light_type="scene light", positional=True,
                                    color=(1.0, 0.99, 0.97), intensity=2.7,
                                    cone_angle=52, exponent=1.2,
                                    attenuation_values=(0.75, 0.0, 0.00007))
        # preserves the source pair's proportions + anchors its fundus to the target
        organ = self.anatomy_meshes["gallbladder"].copy()
        posterior = self.anatomy_meshes["liver"].copy()
        # aligns the anatomical surface with the two established entry ports
        angle = np.radians(-53.2)
        cosine, sine = np.cos(angle), np.sin(angle)
        anatomy_rotation = np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
        organ.points = organ.points @ anatomy_rotation.T + self.rest_target
        posterior.points = posterior.points @ anatomy_rotation.T + self.rest_target
        self.organ_label_point = organ.points[np.argmax(organ.points[:, 2])].copy()
        organ.compute_normals(inplace=True)
        posterior.compute_normals(inplace=True)
        # keeps the registered biliary + vascular branches attached to their source organs
        duct = self.anatomy_meshes["ducts"].copy()
        vascular = self.anatomy_meshes["vessels"].copy().cell_data_to_point_data(pass_cell_data=True)
        for mesh in (duct, vascular):
            mesh.points = mesh.points @ anatomy_rotation.T + self.rest_target
        vascular.point_data["colour"] = np.array([[151, 55, 57], [112, 58, 66]], dtype=np.uint8)[
            np.rint(vascular.point_data["region"]).astype(int)]
        # retains the exact rounded collision envelope for clearance overlays
        vessel_parts = []
        for first, second in zip(self.structure_points, self.structure_points[1:]):
            direction = second - first
            section = pv.Capsule(center=(first + second) / 2, direction=direction,
                                 radius=1, cylinder_length=np.linalg.norm(direction), resolution=24)
            projection = np.clip((section.points - first) @ direction / np.dot(direction, direction),
                                 0.0, 1.0)
            section.point_data["axis"] = first + projection[:, None] * direction
            radial = section.points - section["axis"]
            section.point_data["radial"] = radial / np.linalg.norm(radial, axis=1)[:, None]
            vessel_parts.append(section)
        vessel = vessel_parts[0].merge(vessel_parts[1:])
        self.vessel_axis = vessel["axis"].copy()
        self.vessel_radial = vessel["radial"].copy()
        del vessel.point_data["axis"], vessel.point_data["radial"]
        self.margin_mesh = vessel.copy()
        self.margin_mesh.points = self.vessel_axis + (
            self.structure_radius + self.required_clearance) * self.vessel_radial
        self.margin_actor = self.viewport.add_mesh(
            self.margin_mesh, color="#ce7a7e", opacity=0.045, lighting=False, render=False)
        # samples the original segments without rounding their centreline away from the collision model
        lengths = np.linalg.norm(np.diff(self.structure_points, axis=0), axis=1)
        distances = np.r_[0.0, np.cumsum(lengths)]
        samples = np.unique(np.r_[distances, np.linspace(
            0.0, distances[-1], int(np.ceil(distances[-1])) + 1)])
        centres = np.column_stack([np.interp(samples, distances, self.structure_points[:, axis])
                                   for axis in range(3)])
        tangents = np.gradient(centres, samples, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        normals = centres - self.port
        normals -= np.sum(normals * tangents, axis=1)[:, None] * tangents
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        sides = np.cross(tangents, normals)
        # forms one continuous vessel with a slight inward taper beneath its tissue sleeves
        vessel = pv.PolyData(centres, lines=np.r_[len(centres), np.arange(len(centres))])
        ends = np.exp(-(samples / 3.2) ** 2) + np.exp(-((distances[-1] - samples) / 3.8) ** 2)
        vessel.point_data["radius"] = self.structure_radius * (
            0.98 + 0.02 * np.sin(0.19 * samples + 0.4) ** 2 - 0.07 * ends)
        vessel.point_data["length"] = samples
        vessel = vessel.tube(scalars="radius", absolute=True, n_sides=64,
                             capping=True).triangulate().clean()
        indices = np.searchsorted(samples, vessel["length"])
        radial = vessel.points - centres[indices]
        angle = np.arctan2(np.sum(radial * sides[indices], axis=1),
                           np.sum(radial * normals[indices], axis=1))
        length = vessel["length"].copy()
        coordinates = np.column_stack((length, np.mod(angle, 2 * np.pi) * self.structure_radius))
        # varies the outer wall colour with restrained longitudinal fibres + mottling
        mottling = np.sin(0.7 * length + 1.6 * np.sin(3 * angle))
        fibres = np.exp(-(np.sin(5 * angle + 0.08 * length) / 0.12) ** 2)
        colour = (np.array([163, 69, 75]) + mottling[:, None] * [7, 4, 4]
                  + fibres[:, None] * [12, 10, 8])
        # feathers an uneven pale outer sheath into the fatty ends
        edge = np.minimum(length + 0.8 * np.sin(2 * angle) + 0.4 * np.sin(5 * angle),
                          distances[-1] - length + 1.1 * np.sin(3 * angle + 0.8))
        sheath = np.clip((6 - edge) / 4, 0, 1)
        sheath = 0.85 * sheath * sheath * (3 - 2 * sheath)
        colour = colour * (1 - sheath[:, None]) + sheath[:, None] * [200, 174, 148]
        vessel.clear_data()
        vessel.active_texture_coordinates = coordinates.astype(np.float32)
        vessel.point_data["colour"] = np.rint(colour).astype(np.uint8)
        self.vessel_actor = self.viewport.add_mesh(
            vessel, scalars="colour", rgb=True, show_scalar_bar=False,
            smooth_shading=True, ambient=0.18, pbr=True, metallic=0.0, roughness=0.58, render=False)
        blend = np.sin(np.pi * samples / distances[-1]) ** 2
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        fold_radius = self.structure_radius + self.required_clearance - 0.3
        # beds the vessel in fat + buries both terminal caps within the clearance envelope
        widths = fold_radius * (0.72 + 0.06 * blend + 0.10 * ends
                                + 0.025 * blend * np.sin(0.8 * samples))
        heights = fold_radius * (0.16 + 0.65 * ends)
        centres += (fold_radius * (0.48 - 0.34 * ends))[:, None] * normals
        directions = (heights[:, None, None] * np.cos(angles)[None, :, None] * normals[:, None, :]
                      + widths[:, None, None] * np.sin(angles)[None, :, None] * sides[:, None, :])
        points = centres[:, None, :] + directions
        # closes both ends with the same flattened section
        cap_radius = fold_radius * 0.82
        cap_angles = np.linspace(np.pi / 2, 0, 7)[1:-1]
        caps = []
        for end, sign in ((0, -1), (-1, 1)):
            phase = cap_angles if end == 0 else cap_angles[::-1]
            caps.append(centres[end] + sign * cap_radius * np.sin(phase)[:, None, None]
                        * tangents[end] + np.cos(phase)[:, None, None]
                        * directions[end])
        points = np.concatenate((caps[0], points, caps[1]))
        rings = np.arange(len(points) * len(angles)).reshape(-1, len(angles))
        following = np.roll(rings, -1, axis=1)
        faces = np.stack((np.full_like(rings[:-1], 4), rings[:-1], following[:-1],
                          following[1:], rings[1:]), axis=-1)
        poles = centres[[0, -1]] + np.array([-1, 1])[:, None] * cap_radius * tangents[[0, -1]]
        count = points.size // 3
        fans = np.vstack((np.column_stack((np.full(len(angles), 3),
                                          np.full(len(angles), count), following[0], rings[0])),
                          np.column_stack((np.full(len(angles), 3),
                                           np.full(len(angles), count + 1), rings[-1], following[-1]))))
        tissue = pv.PolyData(np.vstack((points.reshape(-1, 3), poles)),
                             faces=np.r_[faces.ravel(), fans.ravel()]).triangulate().compute_normals(
                                 auto_orient_normals=True, consistent_normals=True)
        tissue.point_data["breathing"] = np.zeros(tissue.n_points)
        # forms a contact plate while leaving the target fundus exposed
        cells, last = posterior.find_closest_cell(organ.points, return_closest_point=True)
        direction = last - organ.points
        local = (organ.points - self.rest_target) @ anatomy_rotation
        contact = (local[:, 0] + 0.45 * np.sin(0.3 * local[:, 1])
                   + 0.25 * np.cos(0.4 * local[:, 2])) > 6
        contact &= np.linalg.norm(direction, axis=1) < 5
        contact &= np.sum(direction * organ.point_normals, axis=1) > -0.2
        triangles = organ.faces.reshape(-1, 4)[:, 1:]
        triangles = triangles[np.all(contact[triangles], axis=1)]
        indices, inverse = np.unique(triangles, return_inverse=True)
        triangles = inverse.reshape(-1, 3)
        first = organ.points[indices] - 0.25 * organ.point_normals[indices]
        # retains surface spacing to prevent collapsed projection cells
        anchors = last[indices]
        normal = posterior.cell_normals[cells[indices]]
        last = 0.8 * anchors + 0.2 * organ.points[indices]
        depth = np.maximum(np.sum((last - anchors) * normal, axis=1), 0) + 0.35
        last -= depth[:, None] * normal
        edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        border = np.unique(edges[counts[inverse] == 1])
        distance = np.linalg.norm(organ.points[indices, None] - organ.points[indices[border]][None], axis=2).min(axis=1)
        blend = np.clip(distance / 2.4, 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        # tapers the exposed border into the gallbladder surface
        last = first + blend[:, None] * (last - first) + (1 - blend[:, None]) * 0.15 * organ.point_normals[indices]
        contacts = [(first, last, triangles, np.ones(len(first)))]
        # follows the liver behind the neck while keeping the registered branches exposed
        neck = organ.points[np.argmax(organ.points[:, 0])]
        distance = np.linalg.norm(posterior.points - neck, axis=1)
        near = np.flatnonzero(distance < 10)
        _, near_duct = duct.find_closest_cell(posterior.points[near], return_closest_point=True)
        _, near_vessel = vascular.find_closest_cell(posterior.points[near], return_closest_point=True)
        gap = np.minimum(np.linalg.norm(near_duct - posterior.points[near], axis=1),
                         np.linalg.norm(near_vessel - posterior.points[near], axis=1))
        score = np.full(posterior.n_points, -1.0)
        score[near] = np.minimum((10 - distance[near]) / 2.4, (gap - 0.8) / 0.8)
        triangles = posterior.faces.reshape(-1, 4)[:, 1:]
        triangles = triangles[np.all(score[triangles] > 0, axis=1)]
        indices, inverse = np.unique(triangles, return_inverse=True)
        triangles = inverse.reshape(-1, 3)
        edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        border = np.unique(edges[counts[inverse] == 1])
        distance = np.linalg.norm(posterior.points[indices, None] - posterior.points[indices[border]][None], axis=2).min(axis=1)
        blend = np.clip(np.minimum(score[indices], distance / 2.4), 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        first = posterior.points[indices] - 0.35 * posterior.point_normals[indices]
        last = posterior.points[indices] + (-0.15 + 0.65 * blend[:, None]) * posterior.point_normals[indices]
        contacts.append((first, last, triangles, np.ones(len(first))))
        # joins a complete body lining to a curved anterior wall with two real port openings
        abdomen = self.anatomy_meshes["abdomen"].copy()
        # shares one curved window boundary between the paired skin + lining surfaces
        count = abdomen.n_points // 2
        middle = (abdomen.points[:count] + abdomen.points[count:]) / 2
        window_direction = np.array([-0.58, -0.35, 0.74])
        window_direction /= np.linalg.norm(window_direction)
        window_right = np.array([1.0, 0.0, 0.0]) - window_direction[0] * window_direction
        window_right /= np.linalg.norm(window_right)
        window_axes = np.array([window_right / 84,
                                np.cross(window_direction, window_right) / 100,
                                window_direction / 65])
        window_centre = np.array([118.0, 12.0, -4.0]) + 20 * (
            window_right - np.cross(window_direction, window_right))
        coordinates = (middle - window_centre) @ window_axes.T
        # keeps the opposite wall + a continuous skin bridge around both entry ports
        entry = middle / [18, 95, 65] - [0, -35 / 95, 15 / 65]
        window = np.maximum(np.sum(coordinates[:, :2] ** 2, axis=1) - 1, -coordinates[:, 2])
        entry = 1 - np.sum(entry ** 2, axis=1)
        window = (window + entry + np.sqrt((window - entry) ** 2 + 0.04 ** 2)) / 2
        abdomen.point_data["section"] = np.tile(window, 2)
        outer = abdomen["region"][abdomen.faces.reshape(-1, 4)[:, 1]] == 2
        body = abdomen.extract_cells(outer).extract_surface(algorithm="dataset_surface")
        cavity = abdomen.extract_cells(~outer).extract_surface(algorithm="dataset_surface")
        boundary = body.copy()
        indices = boundary["vtkOriginalPointIds"]
        boundary.points = middle[indices]
        boundary.point_data["outside"] = abdomen.points[indices]
        boundary.point_data["inside"] = abdomen.points[indices + count]
        boundary = boundary.contour([0.0], scalars="section")
        edges = boundary.lines.reshape(-1, 3)[:, 1:]
        count = boundary.n_points
        depth = np.unique(np.r_[np.linspace(0, 1, 9), 0.12, 0.22, 0.6, 0.78])
        points = ((1 - depth[:, None, None]) * boundary["outside"][None]
                  + depth[:, None, None] * boundary["inside"][None])
        rings = np.arange(len(depth) * count).reshape(-1, count)
        faces = np.stack((np.full((len(depth) - 1, len(edges)), 4),
                          rings[:-1, edges[:, 0]], rings[:-1, edges[:, 1]],
                          rings[1:, edges[:, 1]], rings[1:, edges[:, 0]]), axis=-1).ravel()
        section = pv.PolyData(points.reshape(-1, 3), faces=faces).triangulate()
        section.point_data["depth"] = np.repeat(depth, count)
        # rounds the inspection edge slightly while keeping its skin + lining seams exact
        depth = section["depth"]
        coordinates = (section.points - window_centre) @ window_axes.T
        normal = 2 * coordinates[:, :2] @ window_axes[:2]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        section.points -= 0.35 * np.sin(np.pi * depth)[:, None] ** 2 * normal
        # distinguishes skin, subcutaneous fat + the deeper muscular lining at the cut edge
        layers = np.array([[201, 149, 125], [219, 171, 148], [231, 201, 144],
                           [225, 187, 125], [158, 77, 76], [211, 145, 135]])
        section.point_data["colour"] = np.rint(np.column_stack([
            np.interp(depth, [0, 0.12, 0.22, 0.6, 0.78, 1], layers[:, axis])
            for axis in range(3)])).astype(np.uint8)
        self.section_actor = self.viewport.add_mesh(
            section, scalars="colour", rgb=True, show_scalar_bar=False,
            smooth_shading=True, pbr=True, roughness=0.65, render=False)
        self.section_actor.user_matrix = self.scene_matrix
        attachments = []
        # anchors the vessel bed to its nearest host instead of stretching a ribbon across the cavity
        hosts = (posterior, cavity)
        anchors = [host.find_closest_cell(centres, return_closest_point=True)[1] for host in hosts]
        gaps = np.array([np.linalg.norm(points - centres, axis=1) for points in anchors])
        host_index, nearest = np.unravel_index(np.argmin(gaps), gaps.shape)
        host = hosts[host_index]
        edge = np.linspace(-1, 1, 28)
        first = (centres[nearest] + 0.15 * normals[nearest]
                 + 3.2 * edge[:, None] * sides[nearest])
        direction = anchors[host_index][nearest] - centres[nearest]
        direction /= np.linalg.norm(direction)
        first -= 0.6 * max(0.0, np.dot(direction, normals[nearest])) * normals[nearest]
        guide = anchors[host_index][nearest] + 10 * edge[:, None] * sides[nearest]
        cells, last = host.find_closest_cell(guide, return_closest_point=True)
        last -= 0.65 * host.cell_normals[cells]
        across = np.linspace(0, 1, 38)[None, :, None]
        points = first[:, None, :] * (1 - across) + last[:, None, :] * across
        folds = 0.8 + 0.45 * np.cos(1.5 * np.pi * edge)
        points += folds[:, None, None] * np.sin(np.pi * across) * normals[nearest]
        # broadens the middle into a soft irregular pad while retaining both embedded ends
        points += (1.5 * edge * (1 - edge ** 2))[:, None, None] * np.sin(
            np.pi * across) * sides[nearest]
        weights = np.broadcast_to(across[0, :, 0] if host_index == 0 else 0.0, points.shape[:2])
        attachments.append((points, weights))
        # gives each pad variable thickness + tapered embedded edges
        for points, weights in attachments:
            sheet = pv.StructuredGrid(*np.moveaxis(points, -1, 0)).extract_surface(
                algorithm="dataset_surface").triangulate().compute_normals(consistent_normals=True)
            # follows surface point ordering when assigning the moving edge
            weights = weights.ravel(order="F")[sheet["vtkOriginalPointIds"]]
            original = sheet["vtkOriginalPointIds"]
            width = np.tile(edge, points.shape[1])[original]
            along = np.repeat(np.linspace(0, 1, points.shape[1]), len(edge))[original]
            thickness = 0.22 + 0.9 * (1 - width ** 2) * np.sin(np.pi * along) ** 2
            thickness *= 0.94 + 0.06 * np.cos(7 * width + 5 * along)
            first = sheet.points - 0.5 * thickness[:, None] * sheet.point_normals
            last = sheet.points + 0.5 * thickness[:, None] * sheet.point_normals
            contacts.append((first, last, sheet.faces.reshape(-1, 4)[:, 1:], weights))
        # closes the host-conforming layers with shared geometry + breathing weights
        for first, last, triangles, weights in contacts:
            count = len(first)
            edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
            _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                           return_inverse=True, return_counts=True)
            edges = edges[counts[inverse] == 1]
            triangles = np.vstack((triangles[:, ::-1], triangles + count,
                                   np.column_stack((edges, edges[:, 1] + count)),
                                   np.column_stack((edges[:, 0], edges[:, 1] + count, edges[:, 0] + count))))
            plate = pv.PolyData(np.vstack((first, last)),
                                faces=np.column_stack((np.full(len(triangles), 3), triangles)).ravel())
            plate.point_data["breathing"] = np.tile(weights, 2)
            tissue = tissue.merge(plate, merge_points=False)
        tissue = tissue.clean(tolerance=1e-6, absolute=True)
        tissue.compute_normals(auto_orient_normals=True, consistent_normals=True, inplace=True)
        outline = abdomen.field_data["front_outline"]
        hole_angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        tool_opening = self.wall_opening_radius * np.column_stack((np.cos(hole_angles), np.sin(hole_angles)))
        slope = self.scope_direction[1:] / self.scope_direction[0]
        major = slope / np.linalg.norm(slope)
        minor = np.array([-major[1], major[0]])
        thickness = self.wall_bounds[1] - self.wall_bounds[0]
        radius = self.scope_opening_radius + thickness / 2 * np.linalg.norm(self.scope_direction[1:])
        centre = (self.scope_port - self.port)[1:] + np.mean(self.wall_bounds[:2]) * slope
        scope_opening = (centre + radius / self.scope_direction[0] * np.cos(hole_angles)[:, None] * major
                         + radius * np.sin(hole_angles)[:, None] * minor)
        layers = []
        for layer in range(2):
            points = np.vstack((outline, tool_opening, scope_opening))
            loops = (np.arange(len(outline)), len(outline) + np.arange(64)[::-1],
                     len(outline) + 64 + np.arange(64)[::-1])
            lines = np.vstack([np.column_stack((np.full(len(loop), 2), loop, np.roll(loop, -1)))
                               for loop in loops]).ravel()
            boundary = pv.PolyData(np.column_stack((points, np.zeros(len(points)))), lines=lines)
            cap = boundary.triangulate_contours()
            # closes the small entry patch while preserving both established port pivots
            cap.points = np.column_stack((np.full(cap.n_points, layer * thickness),
                                          cap.points[:, :2])) + self.port
            cap.clear_data()
            cap.point_data["region"] = np.full(cap.n_points, 0 if layer else 2, dtype=np.uint8)
            cap.compute_normals(consistent_normals=True, split_vertices=False, inplace=True)
            normal = cap.cell_normals[:, 0].mean()
            if (layer == 0 and normal > 0) or (layer == 1 and normal < 0):
                cap.flip_faces(inplace=True)
                cap.flip_normal_vectors(inplace=True)
            layers.append(cap)
        openings = np.vstack((tool_opening, scope_opening))
        points = np.vstack((np.column_stack((np.zeros(128), openings)),
                            np.column_stack((np.full(128, thickness), openings)))) + self.port
        edges = np.arange(128).reshape(2, 64)
        next_edges = np.roll(edges, -1, axis=1)
        faces = np.stack((np.full_like(edges, 4), edges, edges + 128,
                          next_edges + 128, next_edges), axis=-1).ravel()
        port_walls = pv.PolyData(points, faces=faces).triangulate()
        port_walls.point_data["region"] = np.ones(port_walls.n_points, dtype=np.uint8)
        for mesh in (*layers, port_walls):
            coordinates = (mesh.points - window_centre) @ window_axes.T
            entry = mesh.points / [18, 95, 65] - [0, -35 / 95, 15 / 65]
            window = np.maximum(np.sum(coordinates[:, :2] ** 2, axis=1) - 1, -coordinates[:, 2])
            entry = 1 - np.sum(entry ** 2, axis=1)
            mesh.point_data["section"] = (window + entry + np.sqrt(
                (window - entry) ** 2 + 0.04 ** 2)) / 2
        wall = layers[0].merge(port_walls, merge_points=False)
        wall.cell_data["colour"] = np.tile([255, 255, 255], (wall.n_cells, 1)).astype(np.uint8)
        wall.cell_data["colour"][-port_walls.n_cells:] = [233, 193, 179]
        cavity = cavity.merge(layers[1], merge_points=False)
        # preserves the digestive group's shared anatomy + moves it with the adjacent liver
        stomach, duodenum, bowel, peritoneum = [self.anatomy_meshes[name].copy()
                                               for name in ("stomach", "duodenum", "bowel", "peritoneum")]
        # renders the refined mesenteric folds once, with their own fat material
        cells = bowel.faces.reshape(-1, 4)[:, 1:]
        bowel = bowel.extract_cells(np.all(bowel["region"][cells] != 1, axis=1)).extract_surface(
            algorithm="dataset_surface")
        for mesh in (stomach, duodenum, bowel, peritoneum):
            mesh.points = mesh.points @ anatomy_rotation.T + self.rest_target

        # tints full-colour textures without multiplying the old baked colours into them
        for mesh in (organ, posterior, tissue, cavity, body, stomach, duodenum, bowel, peritoneum):
            colour = np.tile([250.0, 246.0, 241.0], (mesh.n_points, 1))
            if mesh is organ:
                colour[:] = [130, 215, 120]
            elif mesh is stomach:
                colour[:] = [255, 225, 214]
            elif mesh is duodenum:
                colour[:] = [255, 237, 216]
            if mesh is cavity or mesh is bowel:
                colours = np.array([[250, 246, 241], [255, 244, 173], [255, 255, 255],
                                    [255, 226, 226], [245, 216, 204]])
                colour = colours[mesh["region"]]
            mesh.point_data["colour"] = np.clip(np.rint(colour), 0, 255).astype(np.uint8)
        offset = organ.points - self.target
        angle = np.arctan2(offset[:, 2], offset[:, 1])
        radius = 4 * (1 + 0.1 * np.sin(3 * angle) + 0.06 * np.cos(5 * angle))
        blend = np.exp(-2 * (np.linalg.norm(offset, axis=1) / radius) ** 2)
        organ.point_data["colour"] = np.rint(
            organ["colour"] * (1 - blend[:, None]) + blend[:, None] * [190, 67, 100]).astype(np.uint8)
        (self.organ_actor, self.posterior_actor, self.tissue_actor, self.cavity_actor, self.body_actor,
         self.stomach_actor, self.duodenum_actor, self.bowel_actor, self.peritoneum_actor) = [
            self.viewport.add_mesh(mesh, scalars="colour", rgb=True, show_scalar_bar=False,
                                   smooth_shading=True, pbr=True, metallic=0.0, roughness=0.4, render=False)
            for mesh in (organ, posterior, tissue, cavity, body, stomach, duodenum, bowel, peritoneum)]
        self.duct_actor = self.viewport.add_mesh(
            duct, color="#b6b28b", smooth_shading=True, pbr=True, roughness=0.46, render=False)
        self.vascular_actor = self.viewport.add_mesh(
            vascular, scalars="colour", rgb=True, show_scalar_bar=False,
            smooth_shading=True, pbr=True, metallic=0.0, roughness=0.43, render=False)
        diaphragm = self.anatomy_meshes["diaphragm"].copy()
        diaphragm.points = diaphragm.points @ anatomy_rotation.T + self.rest_target
        self.diaphragm_actor = self.viewport.add_mesh(
            diaphragm, color="#f4bfc0", smooth_shading=True, pbr=True, roughness=0.55, render=False)
        self.wall_actor = self.viewport.add_mesh(
            wall, scalars="colour", rgb=True, show_scalar_bar=False, pbr=True, roughness=0.65, render=False)
        # builds a hollow housing that pivots around the fixed entry point
        port = CreateRevolvedMesh([(-18, 5.8), (-17.5, 7), (-14, 7), (-13, 5.2),
                                   (-11.5, 3.4), (-4, 3.4), (-3.5, 3.1),
                                   (-0.5, 3.1), (0, 2.7), (0.4, 2.2),
                                   (5, 2.2), (5.3, 2.05)], inner_radius=1.9)
        port.cell_data["colour"] = np.tile([76, 117, 133], (port.n_cells, 1)).astype(np.uint8)
        port.cell_data["colour"][port.cell_centers().points[:, 0] < -14] = [108, 151, 164]
        port.cell_data["colour"][port.cell_centers().points[:, 0] > 0] = [167, 180, 189]
        self.port_actor = self.viewport.add_mesh(
            port, scalars="colour", rgb=True, show_scalar_bar=False, smooth_shading=True,
            ambient=0.2, specular=0.3, specular_power=25, render=False)

        # separates the anchored housing from sliding laparoscope
        rear = self.scope_length - self.scope_depth
        scope_profile = [(-32, 7.5), (-31.5, 9.5), (-28, 9.5), (-26.5, 6.7),
                         (-23.5, 3.8), (-4, 3.8), (-0.5, 3.8),
                         (0, 3.5), (0.4, 3.2), (5, 3.2), (5.3, 3.05)]
        scope_port = CreateRevolvedMesh(scope_profile, inner_radius=self.scope_radius + 0.4)
        # keeps engraved depth bands outside the body throughout the scope's insertion range
        shaft_positions = np.sort(np.r_[-rear, self.scope_depth - 1,
                                        (self.scope_depth - np.arange(70, 131, 10)[:, None]
                                         + [-0.12, 0.12]).ravel()])
        shaft = CreateRevolvedMesh(np.column_stack((shaft_positions,
                                                   np.full(len(shaft_positions), self.scope_radius))))
        head_profile = [(-23, 5.1), (-21.5, 8.1), (-19.5, 11.4)]
        for position in (-17, -12, -7):
            head_profile.extend(((position - 0.5, 11.4), (position - 0.2, 10.9),
                                 (position + 0.2, 10.9), (position + 0.5, 11.4)))
        head_profile.extend(((-3.5, 11.4), (-1, 9.2), (2, 5), (4, self.scope_radius)))
        head_profile = np.asarray(head_profile)
        head_profile[:, 0] -= rear
        head = CreateRevolvedMesh(head_profile).clean(tolerance=1e-6, absolute=True).clip_closed_surface(
            normal=(0, 0, -1), origin=(0, 0, 10))
        head.compute_normals(split_vertices=True, feature_angle=50, inplace=True)
        cable = pv.Spline([(-rear - 14, -10, 0), (-rear - 22, -22, -4),
                           (-rear - 31, -34, -12), (-rear - 40, -45, -18)], n_points=32).tube(
                               radius=1.4, n_sides=16)
        collar = pv.Cylinder(center=(self.scope_depth - 0.55, 0, 0), direction=(1, 0, 0),
                             radius=self.scope_radius, height=0.9, resolution=32)
        rim = pv.Disc(center=(self.scope_depth - 0.1, 0, 0), normal=(1, 0, 0),
                      inner=1.65, outer=self.scope_radius, c_res=48).extrude((0.1, 0, 0), capping=True)
        lens = pv.Sphere(radius=1, theta_resolution=48, phi_resolution=33)
        lens.points = lens.points * [0.12, 1.65, 1.65] + [self.scope_depth - 0.12, 0, 0]
        del lens.point_data["Normals"]
        lens.compute_normals(inplace=True)
        illumination = pv.Disc(center=(self.scope_depth - 0.04, 0, 0), normal=(1, 0, 0),
                               inner=1.85, outer=2.15, c_res=64)
        scope_port.cell_data["colour"] = np.tile([76, 117, 133], (scope_port.n_cells, 1)).astype(np.uint8)
        scope_port.cell_data["colour"][scope_port.cell_centers().points[:, 0] > 0] = [167, 180, 189]
        self.scope_port_actor = self.viewport.add_mesh(
            scope_port, scalars="colour", rgb=True, show_scalar_bar=False, smooth_shading=True,
            ambient=0.2, specular=0.45, specular_power=35, render=False)
        self.scope_port_actor.user_matrix = self.scene_matrix @ self.scope_matrix
        scope_parts = []
        for mesh, colour in ((shaft, [176, 188, 197]), (head, [35, 43, 50]),
                             (cable, [36, 43, 50]), (collar.merge(rim, merge_points=False), [190, 201, 210]),
                             (lens, [16, 49, 64]), (illumination, [218, 216, 189])):
            if mesh.point_data.active_normals_name is None:
                mesh.compute_normals(split_vertices=True, feature_angle=50, inplace=True)
            mesh.point_data["Normals"] = mesh.point_normals.copy()
            mesh.cell_data["colour"] = np.tile(colour, (mesh.n_cells, 1)).astype(np.uint8)
            if mesh is shaft:
                distance = self.scope_depth - mesh.cell_centers().points[:, 0]
                engraved = (distance > 69) & (distance < 131) & (np.abs(distance / 10 - np.rint(distance / 10)) < 0.02)
                mesh.cell_data["colour"][engraved] = [46, 57, 64]
            scope_parts.append(mesh)
        # adds tactile buttons + a flexible strain relief behind the optical shaft
        for position in (-13.5, -7.5):
            button = pv.Cylinder(center=(-rear + position, 0, 10.3), direction=(0, 0, 1),
                                 radius=1.9, height=0.6, resolution=32)
            button.cell_data["colour"] = np.tile([94, 137, 153], (button.n_cells, 1)).astype(np.uint8)
            scope_parts.append(button)
        relief_direction = np.array([-8.0, -12.0, -4.0])
        for fraction in np.linspace(0.1, 0.8, 6):
            relief = pv.Cylinder(center=np.array([-rear - 14, -10, 0]) + fraction * relief_direction,
                                 direction=relief_direction, radius=2.2, height=0.8, resolution=24)
            relief.cell_data["colour"] = np.tile([48, 56, 63], (relief.n_cells, 1)).astype(np.uint8)
            scope_parts.append(relief)
        scope = scope_parts[0].merge(scope_parts[1:], merge_points=False)
        scope.points -= [self.scope_depth, 0.0, 0.0]
        self.scope_actor = self.viewport.add_mesh(
            scope, scalars="colour", rgb=True, show_scalar_bar=False, smooth_shading=True,
            ambient=0.2, specular=0.45, specular_power=35, render=False)
        matrix = self.scope_matrix.copy()
        matrix[:3, 3] += self.scope_depth * self.scope_direction
        self.scope_actor.user_matrix = self.scene_matrix @ matrix

        # builds each instrument part once
        self.shaft_actor = self.viewport.add_mesh(
            pv.Cylinder(center=(0.5, 0, 0), direction=(1, 0, 0),
                        radius=self.tool_radius, height=1, resolution=64),
            color="#2c343b", smooth_shading=True, ambient=0.22, specular=0.65, specular_power=65, render=False)
        self.external_shaft_actor = self.viewport.add_mesh(
            pv.Cylinder(center=(-0.5, 0, 0), direction=(1, 0, 0),
                        radius=self.tool_radius, height=1, resolution=64),
            color="#2c343b", smooth_shading=True, ambient=0.22, specular=0.65, specular_power=65, render=False)
        grip = CreateRevolvedMesh([(-9.5, 1.5), (-9, 2.7), (-7.5, 3.6),
                                   (-3.5, 3.6), (-2, 2.7), (-1.5, 1.5)])
        grip.points[:, 1:] *= [0.65, 0.85]
        del grip.point_data["Normals"], grip.cell_data["Normals"]
        grip.compute_normals(inplace=True)
        grip.cell_data["colour"] = np.tile([34, 42, 48], (grip.n_cells, 1)).astype(np.uint8)
        handle_parts = [grip]
        for centre, radius in (((-9, -22), (10, 11)), ((-31, -18), (10.5, 10.5))):
            ring = CreateRevolvedMesh([(-1.1, 1.15), (-0.7, 1.2), (0.7, 1.2), (1.1, 1.15)],
                                      inner_radius=0.78)
            inner = np.linalg.norm(ring.cell_centers().points[:, 1:], axis=1) < 0.8
            ring.cell_data["colour"] = np.tile([31, 38, 43], (ring.n_cells, 1)).astype(np.uint8)
            ring.cell_data["colour"][inner] = [55, 65, 71]
            ring.points = np.column_stack((centre[0] + radius[0] * ring.points[:, 1],
                                          ring.points[:, 0], centre[1] - radius[1] * ring.points[:, 2]))
            del ring.point_data["Normals"], ring.cell_data["Normals"]
            ring.compute_normals(inplace=True)
            handle_parts.append(ring)
        for points in (
                [(-4.5, 0, -1.5), (-3, 0, -4), (-2.8, 0, -7), (-4, 0, -11.6)],
                [(-6, 0, -1), (-12, 0, -3), (-18, 0, -7), (-23, 0, -9)]):
            strut = pv.Spline(points, n_points=48).tube(radius=1.5, n_sides=24, capping=True)
            strut.point_data["Normals"] = strut.point_normals.copy()
            strut.cell_data["colour"] = np.tile([38, 46, 52], (strut.n_cells, 1)).astype(np.uint8)
            handle_parts.append(strut)
        rivet = pv.Cylinder(center=(-5.5, 0, -2), direction=(0, 1, 0),
                            radius=0.85, height=3.8, resolution=32)
        rivet.cell_data["colour"] = np.tile([133, 148, 157], (rivet.n_cells, 1)).astype(np.uint8)
        handle_parts.append(rivet)
        self.handle_actor = self.viewport.add_mesh(
            handle_parts[0].merge(handle_parts[1:], merge_points=False),
            scalars="colour", rgb=True, show_scalar_bar=False, smooth_shading=True,
            ambient=0.25, specular=0.2, specular_power=25, render=False)
        knob = CreateRevolvedMesh([(-4, 2.5), (-3.8, 3.4), (-3.5, 3.8),
                                   (-0.6, 3.8), (-0.3, 3.3), (0, 2.7)], resolution=96)
        angles = np.arctan2(knob.points[:, 2], knob.points[:, 1])
        knob.points[:, 1:] *= (1 - 0.035 * (0.5 + 0.5 * np.cos(24 * angles)))[:, None]
        del knob.point_data["Normals"], knob.cell_data["Normals"]
        knob.compute_normals(inplace=True)
        collar_parts = [knob,
                        pv.Cylinder(center=(-5.5, 0, 0), direction=(0, 1, 0), radius=1.25,
                                    height=5, resolution=24)]
        collar = collar_parts[0].merge(collar_parts[1:], merge_points=False)
        collar.cell_data["colour"] = np.tile([52, 61, 70], (collar.n_cells, 1)).astype(np.uint8)
        collar.cell_data["colour"][collar.cell_centers().points[:, 0] < -4] = [135, 144, 153]
        self.collar_actor = self.viewport.add_mesh(
            collar, scalars="colour", rgb=True, show_scalar_bar=False, smooth_shading=True,
            ambient=0.25, specular=0.35, specular_power=30, render=False)
        # keeps closed jaws inside the existing shaft clearance envelope
        sleeve = CreateRevolvedMesh([(-self.tip_length, self.tool_radius),
                                      (-14.8, self.tool_radius), (-14.6, 1.38),
                                      (-13.9, 1.38), (-13.7, 1.46), (-13.2, 1.46),
                                      (-self.jaw_length, 1.2)])
        sleeve.cell_data["colour"] = np.tile([176, 188, 198], (sleeve.n_cells, 1)).astype(np.uint8)
        hinge = pv.Cylinder(center=(-self.jaw_length + 0.7, 0, 0), direction=(0, 1, 0),
                            radius=0.55, height=2.55, resolution=20)
        hinge.cell_data["colour"] = np.tile([100, 115, 125], (hinge.n_cells, 1)).astype(np.uint8)
        jaw_parts = [sleeve, hinge]
        for side in (-1, 1):
            # supports the hinge pin with two cheeks joined to the distal sleeve
            cheek = pv.PolyData(np.array([[-13.3, side * 1.05, -0.3], [-13.3, side * 1.05, 0.3],
                                          [-12.4, side * 1.05, 0.55], [-11.55, side * 1.05, 0.3],
                                          [-11.55, side * 1.05, -0.3], [-12.4, side * 1.05, -0.55]]),
                                faces=[6, 0, 1, 2, 3, 4, 5]).extrude(
                                    (0, side * 0.23, 0), capping=True).triangulate().compute_normals(
                                        auto_orient_normals=True, consistent_normals=True,
                                        split_vertices=True, feature_angle=50)
            cheek.cell_data["colour"] = np.tile([176, 188, 198], (cheek.n_cells, 1)).astype(np.uint8)
            washer = pv.Disc(center=(-self.jaw_length + 0.7, side * 1.28, 0), normal=(0, side, 0),
                             inner=0.18, outer=0.48, c_res=48).extrude((0, side * 0.06, 0), capping=True)
            washer.cell_data["colour"] = np.tile([202, 211, 218], (washer.n_cells, 1)).astype(np.uint8)
            slot = pv.Cube(center=(-self.jaw_length + 0.7, side * 1.345, 0),
                           x_length=0.58, y_length=0.02, z_length=0.07)
            slot.cell_data["colour"] = np.tile([52, 64, 74], (slot.n_cells, 1)).astype(np.uint8)
            jaw_parts.extend((cheek, washer, slot))
        # tapers two closed blades with flat sides + a small bevel around each edge
        outline = np.array([[-13, 0.08], [-13, 0.38], [-12.2, 0.72], [-10.5, 0.78],
                            [-8, 0.61], [-5, 0.38], [-2.2, 0.16], [0, 0],
                            [-1.1, 0.03], [-4, 0.08], [-8, 0.12], [-11.8, 0.1]])
        outline[:, 0] *= self.jaw_length / 13
        inset = 0.96 * outline + 0.04 * outline.mean(axis=0)
        indices = np.arange(4 * len(outline)).reshape(4, -1)
        faces = np.column_stack((np.full(3 * len(outline), 4), indices[:-1].ravel(),
                                  np.roll(indices[:-1], -1, axis=1).ravel(),
                                  np.roll(indices[1:], -1, axis=1).ravel(), indices[1:].ravel())).ravel()
        faces = np.r_[faces, len(outline), indices[0, ::-1], len(outline), indices[-1]]
        for side in (-1, 1):
            points = np.vstack([np.column_stack((edge[:, 0], np.full(len(edge), side * 0.18 + depth),
                                                  side * edge[:, 1]))
                                for edge, depth in ((inset, -0.15), (outline, -0.1),
                                                    (outline, 0.1), (inset, 0.15))])
            jaw = pv.PolyData(points, faces=faces).triangulate().compute_normals(
                auto_orient_normals=True, consistent_normals=True, split_vertices=True, feature_angle=50)
            jaw.cell_data["colour"] = np.tile([176, 188, 198], (jaw.n_cells, 1)).astype(np.uint8)
            jaw.cell_data["colour"][np.abs(jaw.cell_normals[:, 1]) > 0.95] = [150, 168, 181]
            jaw.cell_data["colour"][np.abs(jaw.cell_normals[:, 1]) < 0.9] = [210, 220, 227]
            jaw_parts.append(jaw)
        # preserves each part's hard-edge normals when assembling the closed tip
        for part in jaw_parts:
            if part.point_data.active_normals_name is None:
                part.compute_normals(split_vertices=True, feature_angle=50, inplace=True)
            part.point_data["Normals"] = part.point_normals.copy()
        self.tip_actor = self.viewport.add_mesh(
            jaw_parts[0].merge(jaw_parts[1:], merge_points=False), scalars="colour", rgb=True, show_scalar_bar=False,
            smooth_shading=True, ambient=0.18, specular=0.7, specular_power=65, render=False)

        # maps material detail in local millimetres so breathing + view changes cannot slide it
        for actor, material, scale, roughness, grain, coat in (
                (self.organ_actor, "tissue", 55.0, 0.4, 0.1, 0.3),
                (self.posterior_actor, "liver", 85.0, 0.36, 0.12, 0.32),
                (self.vessel_actor, "tissue", 35.0, 0.58, 0.08, 0.06),
                (self.tissue_actor, "fat", 35.0, 0.56, 0.1, 0.1),
                (self.duct_actor, "tissue", 35.0, 0.44, 0.08, 0.2),
                (self.vascular_actor, "tissue", 35.0, 0.43, 0.05, 0.2),
                (self.diaphragm_actor, "tissue", 65.0, 0.52, 0.1, 0.12),
                (self.stomach_actor, "tissue", 65.0, 0.48, 0.09, 0.18),
                (self.duodenum_actor, "tissue", 45.0, 0.45, 0.1, 0.2),
                (self.bowel_actor, "tissue", 65.0, 0.5, 0.09, 0.14),
                (self.peritoneum_actor, "fat", 35.0, 0.56, 0.1, 0.1),
                (self.cavity_actor, "tissue", 75.0, 0.5, 0.12, 0.14),
                (self.wall_actor, "skin", 60.0, 0.68, 0.22, 0.0),
                (self.body_actor, "skin", 60.0, 0.68, 0.22, 0.0)):
            # resolves colour selection without rendering each incomplete actor
            actor.mapper.GetInputAlgorithm().Update()
            mesh = actor.mapper.dataset.triangulate()
            coordinates = mesh.points[:, 1:].copy()
            if actor is self.vessel_actor:
                coordinates = mesh.active_texture_coordinates.copy()
            if actor is self.cavity_actor:
                # follows the body lining with continuous material detail
                radius = np.linalg.norm(coordinates, axis=1)
                arc = 187 * np.arctan2(radius, mesh.points[:, 0])
                coordinates *= (arc / np.maximum(radius, 1e-6))[:, None]
            mesh.active_texture_coordinates = (coordinates / scale).astype(np.float32)
            # keeps the cropped torso ends distinct from the curved wall when shading
            split = actor in (self.body_actor, self.cavity_actor)
            # rebuilds normals after the anatomy's shared rotation
            for attributes in (mesh.point_data, mesh.cell_data):
                if "Normals" in attributes:
                    del attributes["Normals"]
            mesh.compute_normals(consistent_normals=True, split_vertices=split,
                                 feature_angle=60, inplace=True)
            tangent_filter = vtkPolyDataTangents()
            tangent_filter.SetInputData(mesh)
            tangent_filter.Update()
            mesh = pv.wrap(tangent_filter.GetOutput())
            tangents = mesh.point_data["Tangents"]
            tangents -= mesh.point_normals * np.sum(tangents * mesh.point_normals, axis=1)[:, None]
            invalid = ~np.isfinite(tangents).all(axis=1) | (np.linalg.norm(tangents, axis=1) < 1e-6)
            # supplies a finite tangent at silhouette vertices where planar UVs collapse
            normals = mesh.point_normals[invalid]
            fallback = np.array([0.0, 1.0, 0.0]) - normals * normals[:, 1, None]
            parallel = np.linalg.norm(fallback, axis=1) < 1e-6
            fallback[parallel] = np.cross(normals[parallel], [1.0, 0.0, 0.0])
            tangents[invalid] = fallback
            tangents /= np.linalg.norm(tangents, axis=1)[:, None]
            actor.mapper.dataset = mesh
            if actor is self.tissue_actor:
                # keeps deforming geometry independent of the colour-selection filter
                actor.mapper.SetInputData(mesh)
            # refreshes the scalar pipeline before caching the moving tissue
            actor.mapper.GetInputAlgorithm().Update()
            actor.mapper.interpolate_before_map = False
            actor.prop.SetBaseColorTexture(self.material_textures[material])
            actor.prop.roughness = roughness
            actor.prop.SetNormalTexture(self.material_normal)
            actor.prop.SetNormalScale(grain)
            actor.prop.SetORMTexture(self.material_roughness)
            actor.prop.SetCoatStrength(coat)
            actor.prop.SetCoatIOR(1.38)
            actor.prop.SetCoatRoughness(0.24 if actor in (self.organ_actor, self.posterior_actor) else 0.3)
        self.rest_tissue_points = self.tissue_actor.mapper.dataset.points.copy()
        self.rest_tissue_tangents = self.tissue_actor.mapper.dataset.point_data["Tangents"].copy()
        # caches fixed topology so only stretching attachments need new shading
        tissue = self.tissue_actor.mapper.dataset
        triangles = tissue.faces.reshape(-1, 4)[:, 1:]
        self.tissue_triangles = triangles.copy()
        self.tissue_deforming_cells = np.flatnonzero(np.ptp(tissue["breathing"][triangles], axis=1) != 0)
        self.tissue_deforming_points = np.unique(triangles[self.tissue_deforming_cells])
        point_index = np.full(tissue.n_points, -1, dtype=int)
        point_index[self.tissue_deforming_points] = np.arange(len(self.tissue_deforming_points))
        self.tissue_normal_cells = np.flatnonzero(np.any(point_index[triangles] >= 0, axis=1))
        point_index = point_index[triangles[self.tissue_normal_cells]].ravel()
        self.tissue_normal_mask = point_index >= 0
        self.tissue_normal_points = point_index[self.tissue_normal_mask]
        self.motion_visual_offset = np.zeros(3)
        # derives the external inspection window from the same fully textured enclosure
        sections = []
        for actor in (self.body_actor, self.cavity_actor, self.wall_actor):
            mesh = actor.mapper.dataset.clip_scalar(scalars="section", value=0.0, invert=False)
            section_actor = self.viewport.add_mesh(
                mesh, scalars="colour", rgb=True, show_scalar_bar=False, smooth_shading=True, render=False)
            section_actor.mapper.GetInputAlgorithm().Update()
            for setting in ("Interpolation", "Metallic", "Roughness", "CoatStrength",
                            "CoatRoughness", "CoatIOR", "NormalScale"):
                getattr(section_actor.prop, f"Set{setting}")(getattr(actor.prop, f"Get{setting}")())
            for slot in ("albedoTex", "normalTex", "materialTex"):
                section_actor.prop.SetTexture(slot, actor.prop.GetTexture(slot))
            section_actor.user_matrix = self.scene_matrix
            sections.append(section_actor)
        self.section_body_actor, self.section_cavity_actor, self.section_wall_actor = sections
        points = np.vstack([actor.mapper.dataset.points for actor in sections]) @ self.scene_rotation.T
        self.section_bounds = np.column_stack((points.min(axis=0), points.max(axis=0))).ravel()
        # separates steel, coated shafts + plastic through their response to light
        for actor, metallic, roughness in (
                (self.shaft_actor, 0.0, 0.5), (self.external_shaft_actor, 0.0, 0.5),
                (self.tip_actor, 0.5, 0.24), (self.scope_actor, 0.15, 0.33),
                (self.collar_actor, 0.12, 0.4), (self.handle_actor, 0.0, 0.6),
                (self.port_actor, 0.05, 0.43), (self.scope_port_actor, 0.05, 0.43)):
            actor.prop.interpolation = "pbr"
            actor.prop.metallic = metallic
            actor.prop.roughness = roughness

        # keeps sensor overlays distinct from the physical instrument
        marker = pv.PolyData(
            np.array([[-3.5, 0, 0], [3.5, 0, 0], [0, -3.5, 0],
                      [0, 3.5, 0], [0, 0, -3.5], [0, 0, 3.5]], dtype=float),
            lines=np.array([2, 0, 1, 2, 2, 3, 2, 4, 5]))
        (self.tool_measurement_actor, self.target_measurement_actor,
         self.tool_estimate_actor, self.target_estimate_actor) = [
            self.viewport.add_mesh(marker, color=color, line_width=width,
                                   render_lines_as_tubes=True, lighting=False, render=False)
            for color, width in [("#287dc0", 2), ("#b050a2", 2), ("#12536e", 3), ("#723d76", 3)]]
        for actor in (self.tool_measurement_actor, self.target_measurement_actor):
            actor.SetVisibility(self.measurements_checkbox.isChecked())
        for actor in (self.tool_estimate_actor, self.target_estimate_actor):
            actor.SetVisibility(self.estimates_checkbox.isChecked())
        # applies the shared layout to anatomy + tracking overlays
        for actor in (self.organ_actor, self.posterior_actor, self.vessel_actor, self.margin_actor, self.cavity_actor,
                      self.tissue_actor, self.duct_actor, self.vascular_actor, self.diaphragm_actor, self.stomach_actor,
                      self.duodenum_actor, self.bowel_actor, self.peritoneum_actor,
                      self.wall_actor, self.body_actor,
                      self.tool_measurement_actor,
                      self.target_measurement_actor, self.tool_estimate_actor,
                      self.target_estimate_actor):
            actor.user_matrix = self.scene_matrix
        self.label_points = pv.PolyData(np.array([
            self.port, self.target, self.structure_points[len(self.structure_points) // 2],
            self.organ_label_point, self.port + [0.0, -22.0, 18.0],
            self.scope_port - (rear + 6) * self.scope_direction]))
        self.label_points.points = self.label_points.points @ self.scene_rotation.T
        self.label_points["labels"] = ["Fixed port", "Target", "Blood vessel", "Organ surface", "Body wall",
                                        "Laparoscope"]
        self.label_actor = self.viewport.add_point_labels(
            self.label_points, "labels",
            font_size=12, text_color="#25384b", point_size=0, shape_opacity=0.0,
            always_visible=True, show_points=False, render=False)
        self.label_actor.SetVisibility(self.labels_checkbox.isChecked())
        for actor in self.viewport.renderer.actors.values():
            if isinstance(actor, pv.Actor):
                actor.mapper.GetInputAlgorithm().Update()
        self.viewport.add_axes(color="#62788c")

    def UpdateRespiratoryMotion(self, now=None, *, visual=True, reset=False):
        now = perf_counter() if now is None else now
        if reset:
            self.stop_path()
            self.pressed_keys.clear()
            self.camera_mouse_position = None
            self.camera_command = self.camera_configuration.copy()
            self.motion_started_time = now
            self.UpdateRespiratoryMotion(now)
            self.reset_measurements()
            self.message_label.setText("Breathing settings updated. Tool and scope poses retained; tracking restarted.")
            return
        if self.motion_started_time is None:
            self.motion_started_time = now
        self.motion_displacement, self.motion_velocity = GetRespiratoryMotion(
            self.motion_amplitude.value(), self.motion_frequency.value(), now - self.motion_started_time)
        offset = np.array([0.0, 0.0, self.motion_displacement])
        self.target = self.rest_target + offset
        if not visual or np.array_equal(offset, self.motion_visual_offset):
            return
        self.motion_visual_offset = offset.copy()
        # translates persistent meshes and labels without changing camera framing
        for actor in (self.organ_actor, self.posterior_actor, self.duct_actor, self.vascular_actor, self.diaphragm_actor,
                      self.stomach_actor, self.duodenum_actor, self.bowel_actor, self.peritoneum_actor):
            actor.SetPosition(*offset)
        # keeps the vessel edge anchored while its tissue attachments follow the liver
        tissue = self.tissue_actor.mapper.dataset
        tissue.points = self.rest_tissue_points + tissue["breathing"][:, None] * offset
        if len(self.tissue_deforming_cells):
            points = tissue.points[self.tissue_triangles[self.tissue_deforming_cells]]
            normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
            length = np.linalg.norm(normals, axis=1)
            normals /= np.maximum(length, np.finfo(float).tiny)[:, None]
            tissue.cell_data["Normals"][self.tissue_deforming_cells] = normals
            # averages incident unit face normals in the same cell order as VTK
            normals = np.zeros((len(self.tissue_deforming_points), 3), dtype=np.float32)
            neighbours = np.repeat(tissue.cell_normals[self.tissue_normal_cells], 3, axis=0)
            np.add.at(normals, self.tissue_normal_points, neighbours[self.tissue_normal_mask])
            normals /= np.maximum(np.linalg.norm(normals, axis=1), np.finfo(np.float32).tiny)[:, None]
            tissue.point_data["Normals"][self.tissue_deforming_points] = normals
            tangents = self.rest_tissue_tangents[self.tissue_deforming_points].copy()
            tangents -= normals * np.sum(tangents * normals, axis=1)[:, None]
            invalid = np.linalg.norm(tangents, axis=1) < 1e-6
            fallback = np.cross(normals[invalid], [1.0, 0.0, 0.0])
            parallel = np.linalg.norm(fallback, axis=1) < 1e-6
            fallback[parallel] = np.cross(normals[invalid][parallel], [0.0, 1.0, 0.0])
            tangents[invalid] = fallback
            tissue.point_data["Tangents"][self.tissue_deforming_points] = (
                tangents / np.linalg.norm(tangents, axis=1)[:, None])
        self.label_points.points[1] = self.scene_rotation @ self.target
        self.label_points.points[3] = self.scene_rotation @ (
            self.organ_label_point + offset)
        self.render_pending = True

    def reset_camera(self, view_index=None):
        # preserves the scope on view changes; Reset view restores its resting pose
        view = self.view_combo.currentText()
        camera_view = view == "Camera view"
        self.pressed_keys.clear()
        self.camera_mouse_position = None
        if not self.following:
            self.command = self.configuration.copy()
        if camera_view and view_index is None:
            self.camera_configuration = np.array([0.0, 0.0, self.scope_depth])
        self.camera_command = self.camera_configuration.copy()
        self.sync_controls()
        self.viewport.camera.parallel_projection = view in ("Top", "Front")
        self.wall_actor.prop.opacity = 1.0
        self.section_actor.SetVisibility(not camera_view)
        for actor in (self.body_actor, self.cavity_actor, self.wall_actor):
            actor.SetVisibility(camera_view)
        for actor in (self.section_body_actor, self.section_cavity_actor, self.section_wall_actor):
            actor.SetVisibility(not camera_view)
        if camera_view:
            self.viewport.remove_all_lights()
            self.viewport.add_light(self.scope_light)
            self.viewport.set_background("#291419", top="#4b272b")
            self.viewport.iren.style = vtkInteractorStyleUser()
            self.viewport.camera.view_angle = self.camera_view_angle
            self.UpdateCameraPose()
            self.viewport.render()
            self.render_pending = False
            self.UpdateSafeSight(reset=True)
            return
        self.viewport.enable_lightkit()
        for light in self.viewport.renderer.lights:
            light.intensity *= 1.5
        self.viewport.add_light(pv.Light(light_type="headlight", intensity=0.45))
        self.viewport.set_background("#edf2f6", top="#fafcfd")
        self.viewport.enable_trackball_style()
        focal = (0, 0, 0)
        cameras = {
            "Perspective": [(-130, -240, 80), self.scene_rotation @ (
                self.port + 0.55 * (self.rest_target - self.port)), (0, 0, 1)],
            "Top": [(0, 0, 300), focal, (0, 1, 0)],
            "Front": [(0, -300, 0), focal, (0, 0, 1)],
            "Overview": [(-130, -240, 80), focal, (0, 0, 1)],
        }
        self.viewport.camera_position = cameras[view]
        self.viewport.camera.view_angle = 30.0
        if view == "Perspective":
            # presents both entry ports side by side, with the laparoscope on the left
            camera = self.viewport.camera
            camera.up = np.cross(np.array(camera.position) - np.array(camera.focal_point),
                                 self.scene_rotation @ (self.port - self.scope_port))
        # fits current physical models without letting sensor overlays shift the view
        actors = (self.shaft_actor, self.external_shaft_actor, self.tip_actor,
                  self.handle_actor, self.collar_actor, self.port_actor, self.section_wall_actor,
                  self.scope_actor, self.scope_port_actor, self.organ_actor, self.posterior_actor,
                  self.vessel_actor, self.tissue_actor, self.duct_actor, self.vascular_actor, self.diaphragm_actor,
                  self.section_cavity_actor, self.section_body_actor, self.stomach_actor, self.duodenum_actor,
                  self.bowel_actor, self.peritoneum_actor)
        bounds = np.array([self.section_bounds if actor in (self.section_body_actor, self.section_cavity_actor, self.section_wall_actor)
                           else actor.GetBounds() for actor in actors
                           if actor.GetVisibility()]).reshape(-1, 3, 2)
        low = bounds[:, :, 0].min(axis=0)
        high = bounds[:, :, 1].max(axis=0)
        if view == "Perspective":
            # frames the torso section + both ports as one anatomical setup
            actors = (self.shaft_actor, self.tip_actor, self.port_actor, self.scope_port_actor,
                      self.organ_actor, self.posterior_actor, self.vessel_actor, self.tissue_actor,
                      self.duct_actor, self.vascular_actor, self.diaphragm_actor,
                      self.section_body_actor, self.section_cavity_actor, self.section_wall_actor,
                      self.stomach_actor, self.duodenum_actor, self.bowel_actor, self.peritoneum_actor)
            bounds = np.array([self.section_bounds if actor in (self.section_body_actor, self.section_cavity_actor, self.section_wall_actor)
                               else actor.GetBounds() for actor in actors
                               if actor.GetVisibility()]).reshape(-1, 3, 2)
            outside = np.vstack((self.scene_rotation @ self.port - 60 * self.scene_rotation
                                  @ InstrumentRotation(*self.configuration[:2])[:, 0],
                                  self.scope_tip - 85 * np.array(self.scope_actor.user_matrix)[:3, 0]))
            low = np.minimum(bounds[:, :, 0].min(axis=0), outside.min(axis=0))
            high = np.maximum(bounds[:, :, 1].max(axis=0), outside.max(axis=0))
        organ_bounds = np.array([actor.mapper.dataset.bounds for actor in (
            self.organ_actor, self.posterior_actor, self.duct_actor, self.vascular_actor, self.diaphragm_actor,
            self.stomach_actor, self.duodenum_actor, self.bowel_actor, self.peritoneum_actor)]).reshape(-1, 3, 2)
        organ_bounds = np.column_stack((organ_bounds[:, :, 0].min(axis=0),
                                       organ_bounds[:, :, 1].max(axis=0)))
        organ_centre = self.scene_rotation @ organ_bounds.mean(axis=1)
        organ_extent = abs(self.scene_rotation) @ (np.diff(organ_bounds, axis=1).ravel() / 2)
        breathing = abs(self.scene_rotation) @ np.array([0.0, 0.0, self.motion_amplitude.value()])
        low = np.minimum(low, organ_centre - organ_extent - breathing)
        high = np.maximum(high, organ_centre + organ_extent + breathing)
        if self.route_points is not None:
            points = self.route_points @ self.scene_rotation.T
            low = np.minimum(low, points.min(axis=0))
            high = np.maximum(high, points.max(axis=0))
        padding = 12 if view == "Perspective" else 8
        low -= padding
        high += padding
        focal = self.scene_rotation @ ((self.port + self.rest_target) / 2)
        if view == "Overview":
            low = np.minimum([-135, -120, -135], low)
            high = np.maximum([135, 120, 170], high)
        elif view in ("Top", "Front"):
            extent = np.maximum(focal - low, high - focal)
            low, high = focal - extent, focal + extent
        bounds = np.column_stack((low, high)).ravel()
        if all(self.viewport.renderer.GetSize()):
            self.viewport.renderer.ResetCameraScreenSpace(bounds, 0.9)
        else:
            self.viewport.reset_camera(bounds=bounds, render=False)
        if view == "Perspective":
            self.viewport.camera.zoom(1.35)
        self.viewport.renderer.ResetCameraClippingRange()
        self.viewport.render()
        self.render_pending = False

        self.UpdateSafeSight(reset=True)

    def RunComparison(self):
        if self.comparison_process is not None:
            return
        if self.planning_future is not None:
            self.comparison_status_label.setText("Finish or cancel route planning first.")
            return
        self.UpdateComparison()
        # captures widget values before sending plain settings to the worker
        axis = self.scene_rotation.T @ np.eye(3)[self.sensor_axis.currentIndex()]
        settings = {
            "tracking": ("ideal", "raw", "filtered")[self.navigation_mode_combo.currentIndex()],
            "tool_noise": self.tool_noise.value(), "target_noise": self.target_noise.value(),
            "tool_dropout": self.tool_dropout.value() / 100,
            "target_dropout": self.target_dropout.value() / 100,
            "tool_outlier": self.tool_outlier.value() / 100,
            "target_outlier": self.target_outlier.value() / 100,
            "outlier_size": self.outlier_size.value(),
            "reject_outliers": self.reject_outliers.isChecked(),
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
        self.comparison_response = None
        self.comparison_progress = self.study_context.Queue()
        self.comparison_results = self.study_context.Queue()
        self.comparison_dialog.hide()
        self.comparison_results_button.hide()
        parameter = self.comparison_type.currentData()
        study = parameter is not None
        if study:
            settings["parameter"] = parameter
        level_count = 2 if parameter == "reject_outliers" else 4
        total = 3 * len(seeds) * (level_count if study else 1)
        self.comparison_status_label.setText(f"Completed 0 / {total} trials.")
        self.comparison_process = self.study_context.Process(
            target=RunStudyWorker, daemon=True,
            args=(self.scenario_combo.currentText(), seeds, settings, self.comparison_cancel,
                  self.comparison_progress, self.comparison_results))
        try:
            self.comparison_process.start()
        except Exception as error:
            self.comparison_process.close()
            self.comparison_process = None
            self.comparison_progress.close()
            self.comparison_results.close()
            self.comparison_status_label.setText(f"Comparison failed: {error}")
            return
        self.comparison_button.setEnabled(False)
        self.comparison_type.setEnabled(False)
        self.comparison_trials.setEnabled(False)
        self.comparison_stop_button.setEnabled(True)
        self.plan_button.setEnabled(False)

    def UpdateComparison(self):
        if self.comparison_process is None:
            seed, count = self.sensor_seed.value(), self.comparison_trials.value()
            parameter = self.comparison_type.currentData()
            study = parameter is not None
            levels_label = ("Tool noise σ: 0, 1, 2, 4 mm" if parameter == "tool_noise"
                            else "Uncertainty k: 0, 0.5, 1, 2")
            level_count = 2 if parameter == "reject_outliers" else 4
            if parameter == "reject_outliers":
                levels_label = "Outlier rejection: off / on"
            self.comparison_button.setText("Run study" if study else "Run comparison")
            self.comparison_stop_button.setText("Stop study" if study else "Stop comparison")
            self.comparison_setup_label.setText(
                f"{self.scenario_combo.currentText()}\n{self.navigation_mode_combo.currentText()}\n"
                f"3 methods × {count} trials" + (f" × {level_count} levels" if study else "")
                + f" · Seeds {seed}–{seed + count - 1}"
                + (f"\n{levels_label}" if study else ""))
            return
        # reads worker progress on the UI thread without touching the live scene
        while True:
            try:
                completed, total, mode, seed = self.comparison_progress.get_nowait()
            except Empty:
                break
            state = "Stopping" if self.comparison_cancel.is_set() else "Completed"
            self.comparison_status_label.setText(f"{state} · {completed} / {total} trials.")
        # receives results before waiting for exit so worker can flush its queue
        if self.comparison_response is None:
            try:
                self.comparison_response = self.comparison_results.get_nowait()
            except Empty:
                if self.comparison_process.is_alive():
                    return
                # checks once more after exit
                try:
                    self.comparison_response = self.comparison_results.get_nowait()
                except Empty:
                    self.comparison_response = (False, "Study worker stopped unexpectedly.")
        if self.comparison_process.is_alive():
            return
        self.comparison_process.join(0)
        self.comparison_process.close()
        self.comparison_process = None
        self.comparison_progress.close()
        self.comparison_results.close()
        self.comparison_button.setEnabled(True)
        self.comparison_type.setEnabled(True)
        self.comparison_trials.setEnabled(True)
        self.comparison_stop_button.setEnabled(False)
        self.plan_button.setEnabled(not self.route_feedback_active)
        success, result = self.comparison_response
        self.comparison_response = None
        if not success:
            self.comparison_status_label.setText(f"Comparison failed: {result}")
            return
        self.comparison_result = result
        study = "comparisons" in result
        comparisons = result["comparisons"] if study else [result]
        completed = sum(len(comparison["trials"]) for comparison in comparisons)
        total = 3 * len(result["seeds"]) * (len(result["levels"]) if study else 1)
        state = "Stopped" if result["cancelled"] else "Completed"
        reused = sum(trial.get("reused", False) for comparison in comparisons for trial in comparison["trials"])
        self.comparison_status_label.setText(
            f"{state} · {completed} / {total} trials."
            + (f" {reused} control results reused." if reused else ""))
        if not completed:
            return
        self.comparison_level.blockSignals(True)
        self.comparison_level.clear()
        for comparison in comparisons:
            count, expected = len(comparison["trials"]), 3 * len(result["seeds"])
            level_label = (f"Tool noise σ {comparison['settings']['tool_noise']:g} mm"
                           if not study or result["parameter"] == "tool_noise"
                           else f"Uncertainty k {comparison['settings']['uncertainty_scale']:g}")
            if study and result["parameter"] == "reject_outliers":
                level_label = f"Outlier rejection {'on' if comparison['settings']['reject_outliers'] else 'off'}"
            self.comparison_level.addItem(
                f"{level_label} · {count} / {expected} trials"
                + (" · partial" if count < expected else ""))
        self.comparison_level.blockSignals(False)
        self.comparison_level.setVisible(study)
        graph_index = self.comparison_tabs.indexOf(self.sensitivity_graph)
        has_graph = study and any(len(comparison["trials"]) == 3 * len(result["seeds"])
                                  for comparison in comparisons)
        if has_graph and graph_index == -1:
            self.comparison_tabs.addTab(self.sensitivity_graph, "Sensitivity graph")
        elif not has_graph and graph_index != -1:
            self.comparison_tabs.removeTab(graph_index)
        self.comparison_tabs.setTabText(0, "Results at selected level" if study else "Results")
        title = ("Tool-noise sensitivity study" if result.get("parameter") == "tool_noise"
                 else "Uncertainty-factor sensitivity study")
        if result.get("parameter") == "reject_outliers":
            title = "Outlier rejection study"
        self.comparison_dialog.setWindowTitle(title if study else "Planner comparison")
        self.ShowComparison()
        if has_graph:
            self.PlotSensitivity()
        self.comparison_results_button.show()
        self.comparison_dialog.show()

    def ShowComparison(self):
        if self.comparison_result is None:
            return
        result = self.comparison_result
        study = "comparisons" in result
        parameter = result.get("parameter")
        graph_visible = self.comparison_tabs.currentWidget() is self.sensitivity_graph
        self.comparison_level.setVisible(study and not graph_visible)
        if study:
            index = self.comparison_level.currentIndex()
            if index < 0:
                return
            result = result["comparisons"][index]
        settings = result["settings"]
        world_offsets = {name: tuple(map(float, self.scene_rotation @ settings[name]))
                         for name in ("tool_bias", "target_bias", "tool_drift", "target_drift")}
        noise_label = f"Noise σ: tool {settings['tool_noise']:g} / target {settings['target_noise']:g} mm"
        uncertainty_label = f"Uncertainty k {settings['uncertainty_scale']:g}"
        rejection_label = f"Rejection {'on' if settings['reject_outliers'] else 'off'}"
        if study and graph_visible and parameter == "tool_noise":
            noise_label = f"Tool noise varies across completed levels · Target noise σ {settings['target_noise']:g} mm"
        elif study and graph_visible and parameter == "uncertainty_scale":
            uncertainty_label = "Uncertainty k varies across completed levels"
        elif study and graph_visible:
            rejection_label = "Rejection off / on"
        self.comparison_result_label.setText(
            f"{result['scenario']} · {settings['tracking'].capitalize()} tracking · "
            f"Seeds {result['seeds'][0]}–{result['seeds'][-1]}\n"
            f"{noise_label} · "
            f"Dropout: tool {100 * settings['tool_dropout']:g} / target {100 * settings['target_dropout']:g}%\n"
            f"Outliers: tool {100 * settings['tool_outlier']:g} / target {100 * settings['target_outlier']:g}%"
            f" · Spike {settings['outlier_size']:g} mm · {rejection_label}\n"
            f"World bias XYZ (mm): tool {world_offsets['tool_bias']} / target {world_offsets['target_bias']}\n"
            f"World drift XYZ (mm/s): tool {world_offsets['tool_drift']} / target {world_offsets['target_drift']}\n"
            f"Breathing: {settings['motion_amplitude']:g} mm / {settings['motion_frequency']:g} Hz · "
            f"Proximity weight {settings['proximity_weight']:g} · {uncertainty_label}"
            + ("\nControl results and their original planning times are reused."
               if any(trial.get("reused", False) for trial in result["trials"]) else ""))
        metrics = self.comparison_metrics
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

    def PlotSensitivity(self):
        result = self.comparison_result
        if result is None or "comparisons" not in result:
            return
        # compares complete levels; partial trial groups remain available in the table
        comparisons = [comparison for comparison in result["comparisons"]
                       if len(comparison["trials"]) == 3 * len(result["seeds"])]
        levels = np.array([comparison["settings"][result["parameter"]] for comparison in comparisons])
        key = self.sensitivity_metric.currentData()
        axes = self.sensitivity_axes
        axes.clear()
        for column, (mode, color) in enumerate(zip(
                ("conventional", "prox_aware", "uncert_aware"),
                ("#286e9f", "#00877b", "#7757b5")), 1):
            means, deviations = [], []
            for comparison in comparisons:
                summary = comparison["summary"][mode]
                if key in ("success_rate", "false_arrival_rate"):
                    mean = 100 * (summary["success_rate"] if key == "success_rate"
                                  else summary["false_arrivals"] / summary["trials"])
                    deviation = None
                else:
                    mean, deviation = (summary["metrics"][key][field] for field in ("mean", "sd"))
                means.append(np.nan if mean is None else mean)
                deviations.append(np.nan if deviation is None else deviation)
            means, deviations = np.asarray(means), np.asarray(deviations)
            if not np.any(np.isfinite(means)):
                continue
            axes.plot(levels, means, "o-", color=color,
                      label=self.comparison_table.horizontalHeaderItem(column).text())
            available = np.isfinite(means) & np.isfinite(deviations)
            if np.any(available):
                axes.errorbar(levels[available], means[available], yerr=deviations[available],
                              fmt="none", color=color, capsize=4)
        axes.set_xlabel("Tool noise σ (mm)" if result["parameter"] == "tool_noise" else "Uncertainty factor k")
        axes.set_ylabel(self.sensitivity_metric.currentText())
        axes.set_title(f"Complete study levels: {len(comparisons)} / {len(result['levels'])}")
        axes.set_xticks(levels)
        if result["parameter"] == "reject_outliers":
            axes.set_xlabel("Outlier rejection")
            axes.set_xticks(levels, ["On" if level else "Off" for level in levels])
        if key in ("success_rate", "false_arrival_rate"):
            axes.set_ylim(0, 100)
        axes.grid(alpha=0.2)
        if axes.lines:
            axes.legend()
        else:
            axes.text(0.5, 0.5, "No available measurements for this metric",
                      ha="center", va="center", transform=axes.transAxes)
        self.sensitivity_canvas.draw_idle()

    def plan_path(self):
        if self.comparison_process is not None:
            self.message_label.setText("Stop the comparison before planning a route.")
            return
        source = self.navigation_mode_combo.currentIndex()
        mode = self.planning_mode_combo.currentData()
        if not self.TrackingAvailable(source, mode, accepted=True):
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
            self.tool_radius, self.structure_points.copy(), self.structure_radius,
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
        self.UpdateGuidanceCard()

    def poll_planning(self):
        if self.planning_future is None or not self.planning_future.done():
            return
        future = self.planning_future
        self.planning_future = None
        self.planning_cancel = None
        self.plan_button.setEnabled(self.comparison_process is None)
        try:
            result = future.result()
        except Exception as error:
            self.cancel_button.setEnabled(False)
            self.margin_label.hide()
            self.route_result_label.hide()
            self.plan_label.setText("Planning failed.")
            self.message_label.setText(f"Planning failed: {error}")
            self.UpdateGuidanceCard()
            return
        self.plan_label.setText(f"Search: {result['time']:.3f} s\n"
                               f"Expanded configurations: {result['expanded']}")
        self.margin_label.setText(f"Planned margin · Base {result['base_clearance']:.2f} mm\n"
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
            self.UpdateGuidanceCard()
            return
        self.path = result["path"]
        self.route_clearance = result["required_clearance"]
        self.execution_clearance = self.route_clearance
        self.route_check_key = None
        self.route_clearance_valid = True
        self.margin_mesh.points = self.vessel_axis + (
            self.structure_radius + self.route_clearance) * self.vessel_radial
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
            point_size=9, render_points_as_spheres=True, lighting=False)
        self.path_actor.user_matrix = self.scene_matrix
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
        self.update_route_guidance()

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
        self.tracking_paused_time = None
        self.tracking_recovery_samples = 0
        self.tracking_recovery_sample = None
        self.route_check_key = None
        self.command = self.configuration.copy()
        self.plan_button.setEnabled(self.comparison_process is None)
        self.follow_button.setEnabled(False)
        self.sync_controls()
        if keep_route and self.path is not None:
            self.manual_guidance = True
            self.update_route_guidance()
            self.message_label.setText("Manual control. The route stays visible as a reference; replan for automatic execution.")
            return
        self.route_clearance = self.required_clearance
        self.execution_clearance = self.required_clearance
        self.route_clearance_valid = True
        if self.margin_actor is not None:
            self.margin_mesh.points = self.vessel_axis + (
                self.structure_radius + self.required_clearance) * self.vessel_radial
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
        self.UpdateGuidanceCard()

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
        help_text = "Tool: W / S  Insert / retract\nA / D  Yaw · Q / E  Pitch"
        if self.view_combo.currentText() == "Camera view":
            help_text += "\nScope: left-drag to aim · C / V  Advance / retract"
        self.keyboard_help.setText(help_text)
        values = np.r_[np.rad2deg(self.command[:2]), self.command[2]]
        lower = np.r_[np.rad2deg(self.lower_limits[:2]), self.lower_limits[2]]
        upper = np.r_[np.rad2deg(self.upper_limits[:2]), self.upper_limits[2]]
        for index, (name, unit) in enumerate([("Yaw", "°"), ("Pitch", "°"),
                                              ("Insertion", "mm")]):
            self.sliders[index].blockSignals(True)
            self.sliders[index].setRange(round(lower[index] * 10), round(upper[index] * 10))
            self.sliders[index].setValue(round(values[index] * 10))
            self.sliders[index].blockSignals(False)
            self.command_labels[index].setText(f"{name}: {values[index]:.1f} {unit}")

    def set_paused(self, paused):
        self.pressed_keys.clear()
        self.camera_mouse_position = None
        self.camera_command = self.camera_configuration.copy()
        self.pause_button.setText("Resume movement" if paused else "Pause movement")
        if self.tracking_paused_time is not None:
            self.message_label.setText("Tracking recovery pending. Movement stays paused.")
        elif self.following:
            self.message_label.setText("Route paused." if paused else "Resuming the planned route.")
        else:
            self.message_label.setText("Movement paused." if paused else "Ready for manual movement.")
        self.update_route_guidance()

    def reset_instrument(self):
        self.stop_path()
        self.pressed_keys.clear()
        self.camera_mouse_position = None
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
            if shaft_clearance(self.port, tip, self.tool_radius, self.structure_points,
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
        # keeps fault draws separate from gaussian noise + instrument resets
        seed = self.sensor_seed.value()
        self.sensor_generator = np.random.default_rng(seed)
        self.dropout_generators = [np.random.default_rng([seed, index]) for index in (1, 2)]
        self.outlier_generators = [np.random.default_rng([seed, index]) for index in (3, 4)]
        self.sensor_drift = np.zeros((2, 3))
        self.measurement_count = 0
        self.tool_filter = None
        self.target_filter = None
        # allows the constant-velocity model to follow peak breathing acceleration
        self.target_acceleration = max(1.0, self.motion_amplitude.value()
                                      * (2 * np.pi * self.motion_frequency.value()) ** 2)
        self.sensor_baseline.setText(
            "Constant-velocity filtering.\n"
            f"Motion σa: tool 20 / target {self.target_acceleration:.2f} mm/s².\n"
            "Errors use true positions for evaluation.\n"
            "Motion uses known configuration + obstacle geometry.\n"
            "The selected tracking source supplies target + guide/arrival feedback.")
        self.tool_measurement = None
        self.target_measurement = None
        self.tool_estimate = None
        self.target_estimate = None
        self.observations = [None, None]
        self.observation_times = [None, None]
        self.filter_observation_times = [None, None]
        self.rejected_observations = [False, False]
        self.rejected_counts = np.zeros(2, dtype=int)
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
        route_active = self.route_feedback_active
        sample_time = perf_counter()
        self.UpdateRespiratoryMotion(sample_time, visual=False)
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
            self.structure_points, self.structure_radius)
        timestep = (max(sample_time - self.last_measurement_time, 1e-6)
                    if self.last_measurement_time is not None else None)
        self.measurement_errors = np.full(2, np.nan)
        self.estimate_errors = np.full(2, np.nan)
        self.rejected_observations = [False, False]
        filters = [self.tool_filter, self.target_filter]
        readings = [self.tool_measurement, self.target_measurement]
        estimates = [None, None]
        # maps the displayed world direction back into simulation coordinates
        direction = self.scene_rotation.T @ np.eye(3)[self.sensor_axis.currentIndex()]
        for index, (position, noise, dropout, bias, drift, acceleration) in enumerate((
                (tool_tip, self.tool_noise.value(), self.tool_dropout.value(),
                 self.tool_bias.value(), self.tool_drift.value(), 20.0),
                (self.sampled_target, self.target_noise.value(), self.target_dropout.value(),
                 self.target_bias.value(), self.target_drift.value(), self.target_acceleration))):
            # accumulates drift while preserving previous offsets when its rate changes
            drift_rate = direction * drift
            biased_position = ApplyBias(position, direction * bias + self.sensor_drift[index],
                                        drift_rate, timestep or 0.0)
            self.sensor_drift[index] += drift_rate * (timestep or 0.0)
            # draws noise every tick so dropout does not change the Gaussian sequence
            measured = measure_position(biased_position, noise, self.sensor_generator)
            measured = ApplyOutlier(measured, (self.tool_outlier, self.target_outlier)[index].value() / 100,
                                    self.outlier_size.value(), self.outlier_generators[index])
            observation = ApplyDropout(measured, dropout / 100, self.dropout_generators[index])
            self.observations[index] = observation
            estimator = filters[index]
            if estimator is not None:
                estimator.predict(timestep)
            if observation is not None:
                # confirms startup before exposing a filtered estimate
                if estimator is None:
                    estimator = KalmanFilter(observation, noise, acceleration_std=acceleration,
                                             confirm_startup=True)
                    accepted = True
                else:
                    accepted = estimator.update(observation, reject_outliers=self.reject_outliers.isChecked())
                self.rejected_observations[index] = not accepted
                self.rejected_counts[index] += not accepted
                if accepted and estimator.ready:
                    self.filter_observation_times[index] = sample_time
                readings[index] = observation
                self.observation_times[index] = sample_time
                self.measurement_errors[index] = np.linalg.norm(observation - position)
                self.measurement_squared_error[index] += self.measurement_errors[index] ** 2
                self.measurement_counts[index] += 1
            filters[index] = estimator
            if estimator is not None and estimator.ready:
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
                (estimates, [estimator.covariance[:3, :3] if estimator is not None and estimator.ready else None
                             for estimator in filters])):
            self.target_feedbacks.append(
                get_target_feedback(*positions, self.target_tolerance, *covariances)
                if all(position is not None for position in positions)
                else {"distance": np.nan, "arrived": False, "uncertainty": np.nan})
        self.distance_history.append((sample_time - self.distance_started_time,
                                      *(feedback["distance"] for feedback in self.target_feedbacks)))
        self.CheckRouteTracking()
        route_stopped = route_active and not self.route_feedback_active
        for index, feedback in enumerate(self.target_feedbacks):
            if (not feedback["arrived"] or not self.TrackingAvailable(index)
                    or self.tracking_paused_time is not None
                    or self.route_feedback_active and not self.route_clearance_valid
                    or route_stopped
                    or index == 2 and any(self.rejected_observations)):
                self.arrival_started[index] = None
            elif self.arrival_started[index] is None:
                self.arrival_started[index] = sample_time
            self.arrival_confirmed[index] = (self.arrival_started[index] is not None
                                             and sample_time - self.arrival_started[index] >= self.arrival_dwell - 1e-9)

        if (self.route_feedback_active and self.tracking_paused_time is None
                and self.route_clearance_valid):
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
                self.plan_button.setEnabled(self.comparison_process is None)
                if reported:
                    outcome = "Verified arrival" if actual["arrived"] else "False arrival"
                else:
                    outcome = "Arrival missed" if actual["arrived"] else "Arrival not confirmed"
                self.route_result_label.setText(
                    f"{outcome} · {self.plan_source_name}\nActual distance {actual['distance']:.2f} mm")
                self.message_label.setText(f"{outcome} · actual tip–target distance {actual['distance']:.2f} mm. Plan again to use the latest target estimate.")
                self.UpdateGuidanceCard()
        self.update_route_guidance(refresh=False)
        if self.measurements_checkbox.isChecked() or self.estimates_checkbox.isChecked():
            self.render_pending = True

    def update_sensor_readouts(self):
        # refreshes one snapshot while sampling + filtering continue at 10 Hz
        self.motion_label.setText(f"Actual world Y offset {-self.sampled_motion[0]:.2f} mm\n"
                                  f"Actual world Y velocity {-self.sampled_motion[1]:.2f} mm/s")
        self.toggle_measurements(None, render=False)
        if self.measurements_checkbox.isChecked():
            self.render_pending = True
        self.update_route_guidance()
        rows = ""
        for index, (name, feedback) in enumerate(zip(("Actual", "Raw", "Filtered"), self.target_feedbacks)):
            available = self.TrackingAvailable(index)
            state = ("Within" if feedback["arrived"] else "Outside") if available else "Unavailable"
            if index == 2 and any(self.rejected_observations):
                state = "Prediction"
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
        starting = mode == 2 and any(estimator is not None and not estimator.ready
                                    for estimator in (self.tool_filter, self.target_filter))
        if self.route_result is not None:
            self.feedback_status_label.setText(self.route_result_label.text())
        elif self.tracking_paused_time is not None:
            self.feedback_status_label.setText(
                f"Tracking paused · recovery {self.tracking_recovery_samples} / 2.\nArrival confirmation restarts after recovery.")
        elif self.route_feedback_active and not self.route_clearance_valid:
            self.feedback_status_label.setText("Remaining route lacks updated clearance. Arrival confirmation paused.")
        elif starting:
            self.feedback_status_label.setText("Checking filter startup. Waiting for three consistent tool + target readings.")
        elif not self.TrackingAvailable(mode):
            self.feedback_status_label.setText("Tracking unavailable. Arrival confirmation requires fresh tool + target readings.")
        elif mode == 2 and any(self.rejected_observations):
            self.feedback_status_label.setText("Prediction; awaiting accepted tool + target readings.")
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
            received = (self.filter_observation_times if filtered else self.observation_times)[index]
            fresh = (received is not None and self.observations[index] is not None
                     and now - received <= self.tracking_timeout)
            status = "Waiting for first reading" if received is None else (
                f"Fresh · age {now - received:.1f} s" if fresh else
                f"{'Prediction' if filtered else 'Last reading'} · age {now - received:.1f} s")
            estimator = (self.tool_filter, self.target_filter)[index]
            if estimator is not None and not estimator.ready:
                if filtered:
                    status = "Checking startup · waiting for 3 consistent readings"
                elif self.rejected_observations[index]:
                    status += " · startup inconsistent"
            elif self.rejected_observations[index]:
                status = (f"Prediction · accepted age {now - received:.1f} s" if filtered
                          else f"{'Fresh' if fresh else 'Last reading'} · rejected by filter")
            position = self.scene_rotation @ position if position is not None else (None,) * 3
            values = "".join(f'<td align="right">{FormatReading(value)}</td>' for value in position)
            label.setText(f'<b>{name}</b><br>{status}<table width="100%" cellspacing="4">'
                          f'<tr style="color:#62788c">{headings}</tr>'
                          f'<tr style="font-family:Consolas">{values}</tr></table>')
        self.measurement_error_label.setText(
            f"Raw error · mm\nTool {FormatReading(self.measurement_errors[0])}   Target {FormatReading(self.measurement_errors[1])}")
        self.estimate_error_label.setText(
            f"Filtered error · mm\nTool {FormatReading(self.estimate_errors[0])}   Target {FormatReading(self.estimate_errors[1])}")
        # calculates model RMS uncertainty from the three position variances
        uncertainties = [np.sqrt(max(0.0, np.trace(estimator.covariance[:3, :3])))
                         if estimator is not None and estimator.ready else None
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
        self.rejected_label.setText(f"Rejected: tool {self.rejected_counts[0]} / target {self.rejected_counts[1]}")
        yaw, pitch = np.rad2deg(self.sampled_configuration[:2])
        self.position_label.setText(f"ACTUAL INSTRUMENT\nYaw {yaw:.1f}° · Pitch {pitch:.1f}° · Depth {self.sampled_configuration[2]:.1f} mm")
        distance = selected["distance"] if mode != 1 or self.TrackingAvailable(mode) else None
        prediction_note = ""
        if mode == 2 and not self.TrackingAvailable(mode):
            prediction_note = (" · Checking startup" if starting else
                               " · Prediction" if np.isfinite(distance) else " · Unavailable")
        self.target_label.setText(f"{self.navigation_mode_combo.currentText().upper()} DISTANCE\n{FormatReading(distance)} mm{prediction_note}  ·  Actual {self.target_feedbacks[0]['distance']:.2f} mm")
        minimum = self.execution_clearance if self.route_feedback_active else self.required_clearance
        guide_margin = (f"\nGuide planned at {self.route_clearance:.2f} mm"
                        if self.path is not None and not self.following else "")
        self.clearance_label.setText(f"SHAFT CLEARANCE\n{self.sampled_clearance:.2f} mm  ·  Movement minimum {minimum:.2f} mm{guide_margin}")
        if self.path is None:
            self.UpdateGuidanceCard()

    def TrackingAvailable(self, source, mode="conventional", *, accepted=False):
        required = (0, 1) if source != 0 else (0,) if mode == "uncert_aware" else ()
        now = perf_counter()
        for index in required:
            filtered = source == 2 or mode == "uncert_aware" and index == 0
            estimator = (self.tool_filter, self.target_filter)[index]
            if filtered and (estimator is None or not estimator.ready):
                return False
            received = (self.filter_observation_times if filtered else self.observation_times)[index]
            if (self.observations[index] is None or received is None
                    or now - received > self.tracking_timeout
                    or accepted and filtered and self.rejected_observations[index]):
                return False
        return True

    def PauseForTrackingLoss(self, now):
        if self.tracking_paused_time is None:
            self.tracking_paused_time = now
            self.message_label.setText("Tracking lost. Movement paused for up to 1 s; waiting for two fresh samples.")
            self.UpdateGuidanceCard()
        self.tracking_recovery_samples = 0
        self.tracking_recovery_sample = self.measurement_count
        self.arrival_started = [None, None, None]
        self.arrival_confirmed = [False, False, False]

    def ResumeAfterTrackingRecovery(self, now):
        # counts sensor samples once even when several frame updates check tracking
        if self.measurement_count == self.tracking_recovery_sample:
            return False
        self.tracking_recovery_sample = self.measurement_count
        self.tracking_recovery_samples += 1
        if self.tracking_recovery_samples < 2:
            self.message_label.setText("Tracking recovering · 1 / 2 fresh samples. Movement stays paused.")
            return False
        if not self.CheckRouteValidity(immediate=True):
            return False
        # excludes the interruption from arrival checking time after movement ends
        if self.route_finished_time is not None:
            self.route_finished_time += now - self.tracking_paused_time
        self.tracking_paused_time = None
        self.tracking_recovery_samples = 0
        self.tracking_recovery_sample = None
        self.message_label.setText(
            "Tracking recovered. Movement remains paused." if self.pause_button.isChecked() else
            "Tracking recovered. Following the route." if self.following else
            "Tracking recovered. Checking arrival feedback.")
        self.update_route_guidance()
        return True

    def CheckRouteTracking(self):
        if self.path is None or self.route_result is not None:
            return True
        available = self.TrackingAvailable(self.plan_source_index, self.plan_mode, accepted=True)
        if not available:
            self.guidance_warning_pending = None
            self.guidance_warning_started = None
        if not self.route_feedback_active:
            valid = self.CheckRouteValidity() if available else False
            if not self.manual_guidance:
                self.follow_button.setEnabled(valid)
            return valid
        now = perf_counter()
        if self.tracking_paused_time is not None and now - self.tracking_paused_time > 1.0 + 1e-9:
            self.guidance_warning = "Tracking did not recover within 1 s."
            self.stop_path(keep_route=True)
            self.message_label.setText("Tracking did not recover within 1 s. Plan a fresh route after tracking returns; the guide stays visible.")
            return False
        if not available:
            self.PauseForTrackingLoss(now)
            return False
        if self.tracking_paused_time is not None:
            return self.ResumeAfterTrackingRecovery(now)
        return self.CheckRouteValidity()

    def CheckRouteValidity(self, *, immediate=False):
        if self.path is None or self.route_result is not None:
            return True
        if self.guidance_warning is not None:
            return False
        if not self.TrackingAvailable(self.plan_source_index, self.plan_mode, accepted=True):
            self.guidance_warning_pending = None
            self.guidance_warning_started = None
            return False
        target = (self.sampled_target, self.target_measurement,
                  self.target_estimate)[self.plan_source_index]
        reason = None
        if np.linalg.norm(target - self.planned_target) > self.target_tolerance + 1e-9:
            reason = "Target tracking differs from the snapshot."
        clearance = self.route_clearance
        if self.plan_mode == "uncert_aware":
            offset = self.tool_estimate - tip_position(self.port, *self.sampled_configuration)
            margin = get_uncertainty_margin(offset, self.tool_filter.covariance[:3, :3],
                                            self.plan_uncertainty_scale)
            clearance = max(clearance, self.required_clearance + margin)
        self.execution_clearance = clearance
        if immediate or clearance > self.route_clearance:
            key = (self.measurement_count, self.path_index, clearance)
            if immediate or key != self.route_check_key:
                self.route_clearance_valid = CheckRemainingRoute(
                    self.port, self.configuration, self.path, self.path_index,
                    self.tool_radius, self.structure_points, self.structure_radius, clearance)
                self.route_check_key = key
        else:
            self.route_clearance_valid = True
            self.route_check_key = None
        if not self.route_clearance_valid and reason is None:
            reason = "Remaining route lacks updated clearance."
        if not immediate:
            if reason != self.guidance_warning_pending:
                self.guidance_warning_pending = reason
                self.guidance_warning_started = self.last_measurement_time if reason is not None else None
            if (reason is None
                    or self.last_measurement_time - self.guidance_warning_started < 0.5 - 1e-9):
                return self.route_clearance_valid
        if reason is None:
            return True
        self.guidance_warning = reason
        self.arrival_started = [None, None, None]
        self.arrival_confirmed = [False, False, False]
        if self.route_feedback_active:
            self.stop_path(keep_route=True)
            self.message_label.setText(f"Route stopped. {reason} Plan a fresh route; the guide stays visible.")
        else:
            self.follow_button.setEnabled(False)
        return False

    def update_route_guidance(self, refresh=True):
        if self.route_points is None:
            if refresh:
                self.UpdateGuidanceCard()
            return
        self.CheckRouteTracking()
        available = self.TrackingAvailable(self.plan_source_index, self.plan_mode, accepted=True)
        guidance = None
        if not self.route_feedback_active:
            guidance = GetRouteGuidance(self.configuration, self.path, self.path_index)
            self.path_index = guidance["waypoint"]
        if not refresh:
            return
        if self.route_result is not None:
            self.UpdateGuidanceCard()
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
        if self.tracking_paused_time is not None:
            status = f"Tracking paused · recovery {self.tracking_recovery_samples} / 2 samples."
        elif self.guidance_warning is not None:
            status = f"Replan advised: {self.guidance_warning}"
        elif self.pause_button.isChecked():
            status = "Movement paused."
        elif not self.route_clearance_valid:
            status = "Updated clearance does not fit the remaining route; automatic movement paused."
        elif guidance is not None:
            if not available:
                status = "Tracking unavailable; movement guidance paused."
            elif self.guidance_warning_pending is not None:
                status = "Tracking changed; movement guidance paused."
            elif guidance["complete"]:
                status = "Waypoints reached; check target feedback."
            elif not movement_is_clear(
                    self.port, self.configuration, self.path[self.path_index], self.tool_radius,
                    self.structure_points, self.structure_radius, self.execution_clearance):
                status = "Next waypoint blocked; plan a fresh route."
            else:
                yaw, pitch = np.rad2deg(guidance["change"][:2])
                moves = [f"{name} {value:+.1f}°" for name, value in (("Yaw", yaw), ("Pitch", pitch))
                         if round(value, 1) != 0]
                depth = guidance["change"][2]
                if round(depth, 1) != 0:
                    moves.append(f"{'Insert' if depth > 0 else 'Retract'} {abs(depth):.1f} mm")
                status = f"Waypoint {self.path_index} / {len(self.path) - 1} · " + " · ".join(moves)
        if self.plan_source_index == 2 and not available:
            status += " Tip-to-guide uses prediction."
        if self.manual_guidance and self.sampled_clearance < self.execution_clearance:
            status += " Below guide clearance; manual minimum still applies."
        self.route_result_label.setText(
            f"{self.plan_source_name} · Tip-to-guide {FormatReading(distance)} mm\n{status}")
        self.UpdateGuidanceCard("Blocked" if status.startswith("Next waypoint blocked") else None)
        return float(distance) if distance is not None else None

    def toggle_measurements(self, _visible, *, render=True):
        now = perf_counter()
        for index, actor in enumerate((self.tool_measurement_actor, self.target_measurement_actor)):
            actor.SetVisibility(self.measurements_checkbox.isChecked()
                                and self.observations[index] is not None
                                and now - self.observation_times[index] <= self.tracking_timeout)
        for actor, estimator in ((self.tool_estimate_actor, self.tool_filter),
                                 (self.target_estimate_actor, self.target_filter)):
            actor.SetVisibility(self.estimates_checkbox.isChecked() and estimator is not None and estimator.ready)
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
            camera_control = self.view_combo.currentText() == "Camera view"
            tool_keys = self.pressed_keys & self.key_directions.keys()
            if tool_keys and not self.pause_button.isChecked():
                velocity = np.zeros(3)
                for key in tool_keys:
                    axis, direction = self.key_directions[key]
                    velocity[axis] += direction * self.speed_limits[axis]
                self.command = np.clip(self.configuration + velocity * timestep,
                                       self.lower_limits, self.upper_limits)
                self.sync_controls()
            if camera_control and not self.pause_button.isChecked():
                direction = sum(self.camera_key_directions[key]
                                for key in self.pressed_keys & self.camera_key_directions.keys())
                if direction:
                    self.camera_command[2] = np.clip(
                        self.camera_configuration[2] + direction * self.camera_speed_limits[2] * timestep,
                        self.camera_lower_limits[2], self.camera_upper_limits[2])
                change = np.clip(self.camera_command - self.camera_configuration,
                                 -self.camera_speed_limits * timestep, self.camera_speed_limits * timestep)
                if np.any(change):
                    self.camera_configuration += change
                    self.UpdateCameraPose()
            if self.route_feedback_active and not self.CheckRouteTracking():
                return
            if self.pause_button.isChecked() or self.planning_future is not None:
                return

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
            clearance = self.execution_clearance if self.following else self.required_clearance
            if movement_is_clear(self.port, self.configuration, proposed, self.tool_radius,
                                 self.structure_points, self.structure_radius,
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
                    self.update_route_guidance(refresh=False)
                    self.message_label.setText("Manual movement · shaft clearance maintained.")
            else:
                if self.following:
                    self.guidance_warning = "Remaining movement lacks required clearance."
                    self.stop_path(keep_route=True)
                    self.message_label.setText("Route stopped: the required shaft clearance cannot be assured.")
                    self.UpdateGuidanceCard("Blocked")
                    return
                self.command = self.configuration.copy()
                self.sync_controls()
                self.message_label.setText("Movement blocked: the required shaft clearance cannot be assured.")
                self.UpdateGuidanceCard("Blocked")
        finally:
            if self.render_pending:
                self.viewport.render()
                self.render_pending = False

    def update_instrument(self):
        yaw, pitch, depth = self.configuration
        # rotates local +X onto the instrument direction
        rotation = InstrumentRotation(yaw, pitch)
        matrix = np.eye(4)
        internal_length = max(depth - self.tip_length, 0)
        matrix[:3, :3] = rotation @ np.diag([internal_length, 1, 1])
        matrix[:3, 3] = self.port
        self.shaft_actor.user_matrix = self.scene_matrix @ matrix
        self.shaft_actor.SetVisibility(bool(internal_length > 0))
        # moves the external handle towards the port as insertion increases
        external_length = self.shaft_length - depth
        external_front = min(depth - self.tip_length, 0)
        matrix[:3, :3] = rotation @ np.diag([external_length + external_front, 1, 1])
        matrix[:3, 3] = self.port + external_front * rotation[:, 0]
        self.external_shaft_actor.user_matrix = self.scene_matrix @ matrix
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.port
        self.port_actor.user_matrix = self.scene_matrix @ matrix
        matrix[:3, 3] = self.port - external_length * rotation[:, 0]
        for actor in (self.handle_actor, self.collar_actor):
            actor.user_matrix = self.scene_matrix @ matrix
        matrix[:3, 3] = tip_position(self.port, *self.configuration)
        self.tip_actor.user_matrix = self.scene_matrix @ matrix
        self.render_pending = True

    def UpdateCameraPose(self):
        # pivots at the fixed port + keeps the first-person eye on the lens
        # maps positive camera yaw onto screen-right
        yaw, pitch, depth = self.camera_configuration
        rotation = self.scene_rotation @ self.scope_matrix[:3, :3] @ InstrumentRotation(-yaw, pitch)
        self.scope_tip = self.scene_rotation @ self.scope_port + depth * rotation[:, 0]
        self.scope_light.position = self.scope_tip
        self.scope_light.focal_point = self.scope_tip + rotation[:, 0]
        matrix = self.scope_matrix.copy()
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.scene_rotation @ self.scope_port
        self.scope_port_actor.user_matrix = matrix
        matrix[:3, 3] = self.scope_tip
        self.scope_actor.user_matrix = matrix
        self.label_points.points[5] = self.scope_tip - (self.scope_length + 6) * rotation[:, 0]
        if self.view_combo.currentText() == "Camera view":
            camera = self.viewport.camera
            camera.SetPosition(*self.scope_tip)
            camera.SetFocalPoint(*(self.scope_tip + rotation[:, 0]))
            camera.SetViewUp(*rotation[:, 2])
            camera.SetClippingRange(*self.camera_clipping_range)
        self.render_pending = True

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.ApplicationDeactivate or (
                watched is self and event.type() == QEvent.Type.WindowDeactivate):
            self.pressed_keys.clear()
            self.camera_mouse_position = None
            self.camera_command = self.camera_configuration.copy()
        in_window = isinstance(watched, QWidget) and watched.window() is self
        viewport_event = in_window and (watched is self.viewport or self.viewport.isAncestorOf(watched))
        camera_control = self.view_combo.currentText() == "Camera view"
        if viewport_event and event.type() in (QEvent.Type.Leave, QEvent.Type.FocusOut):
            self.camera_mouse_position = None
            self.camera_command[:2] = self.camera_configuration[:2]
        if camera_control and viewport_event and event.type() in (
                QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick,
                QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease, QEvent.Type.Wheel):
            if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
                if event.button() == Qt.MouseButton.LeftButton and not self.pause_button.isChecked():
                    watched.setFocus(Qt.FocusReason.MouseFocusReason)
                    self.camera_mouse_position = event.globalPosition()
            elif event.type() == QEvent.Type.MouseMove and self.camera_mouse_position is not None:
                if event.buttons() & Qt.MouseButton.LeftButton and not self.pause_button.isChecked():
                    position = event.globalPosition()
                    delta = position - self.camera_mouse_position
                    # applies mouse aim directly so quick drags remain responsive
                    self.camera_command[:2] = np.clip(
                        self.camera_configuration[:2] + self.camera_mouse_sensitivity * np.array([delta.x(), -delta.y()]),
                        self.camera_lower_limits[:2], self.camera_upper_limits[:2])
                    self.camera_configuration[:2] = self.camera_command[:2]
                    self.UpdateCameraPose()
                    self.camera_mouse_position = position
                else:
                    self.camera_mouse_position = None
                    self.camera_command[:2] = self.camera_configuration[:2]
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self.camera_mouse_position = None
                self.camera_command[:2] = self.camera_configuration[:2]
            event.accept()
            return True
        if in_window and event.type() in (
                QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            tool_key = event.key() in self.key_directions
            if tool_key or (camera_control and event.key() in self.camera_key_directions):
                if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
                    if event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                        self.pressed_keys.discard(event.key())
                    return super().eventFilter(watched, event)
                if event.type() == QEvent.Type.ShortcutOverride:
                    event.accept()
                elif not event.isAutoRepeat():
                    if event.type() == QEvent.Type.KeyPress and not self.pause_button.isChecked():
                        if tool_key:
                            self.start_manual_control()
                        self.pressed_keys.add(event.key())
                    else:
                        self.pressed_keys.discard(event.key())
                return True
            if viewport_event and not camera_control and (
                    event.key() == Qt.Key.Key_R and not event.modifiers() & (
                        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
    
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    self.reset_camera()
                event.accept()
                return True
            if camera_control and viewport_event:
                # prevents VTK shortcuts from changing the scope's physical view
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        self.timer.stop()
        self.sensor_timer.stop()
        self.readout_timer.stop()
        self.capture_timer.stop()
        self.sight_timer.stop()
        if self.calibration_dialog is not None:
            self.calibration_dialog.close()
            self.calibration_canvas.close()
        self.perception_cancel.set()
        if self.perception_future is not None:
            self.perception_future.cancel()
        if self.perception_executor is not None:
            self.perception_executor.shutdown(wait=False, cancel_futures=True)
        if self.capture_future is not None:
            self.capture_future.cancel()
        if self.capture_executor is not None:
            self.capture_executor.shutdown(wait=False, cancel_futures=True)
        if self.planning_cancel is not None:
            self.planning_cancel.set()
        self.comparison_cancel.set()
        if self.comparison_process is not None:
            self.comparison_process.join(0.2)
            if self.comparison_process.is_alive():
                self.comparison_process.terminate()
                self.comparison_process.join()
            self.comparison_process.close()
            self.comparison_process = None
        if self.comparison_progress is not None:
            self.comparison_progress.close()
            self.comparison_results.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        QApplication.instance().removeEventFilter(self)
        self.viewport.close()
        self.distance_canvas.close()
        self.sensitivity_canvas.close()
        super().closeEvent(event)
