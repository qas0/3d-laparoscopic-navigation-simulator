from time import perf_counter

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from geometry import movement_is_clear, shaft_clearance, tip_position


class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Laparoscopic Navigation Simulator")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 650)

        self.port = np.zeros(3)
        self.target = np.array([90.0, 0.0, 0.0])
        self.structure_centre = np.array([65.0, 11.0, 0.0])
        self.structure_radius = 6.0
        self.tool_radius = 1.5
        self.required_clearance = 2.0
        self.configuration = np.array([0.0, 0.0, 35.0])
        self.command = self.configuration.copy()
        self.lower_limits = np.array([np.deg2rad(-45), np.deg2rad(-35), 10.0])
        self.upper_limits = np.array([np.deg2rad(45), np.deg2rad(35), 110.0])
        self.speed_limits = np.array([np.deg2rad(20), np.deg2rad(20), 15.0])
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
        layout.addWidget(QLabel("Static anatomy · Manual control · Distances in millimetres"))
        body = QHBoxLayout()
        layout.addLayout(body, 1)

        controls = QFrame()
        controls.setObjectName("controls")
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_scroll = QScrollArea()
        control_scroll.setFixedWidth(320)
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setWidget(controls)
        body.addWidget(control_scroll)

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

        keyboard_help = QLabel("Hold W / S: insert / retract\n"
                               "Hold A / D: decrease / increase yaw\n"
                               "Hold Q / E: decrease / increase pitch\n\n"
                               "Mouse drag: orbit   ·   Wheel: zoom")
        keyboard_help.setWordWrap(True)
        control_layout.addWidget(keyboard_help)

        self.pause_button = QPushButton("Pause movement")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.set_paused)
        control_layout.addWidget(self.pause_button)
        reset_button = QPushButton("Reset instrument")
        reset_button.clicked.connect(self.reset_instrument)
        control_layout.addWidget(reset_button)
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

        self.viewport = QtInteractor(central, auto_update=False)
        body.addWidget(self.viewport.interactor, 1)
        self.viewport.set_background("#f2f6f9")
        self.build_scene()

        self.position_label = QLabel()
        self.target_label = QLabel()
        self.clearance_label = QLabel()
        readouts = QHBoxLayout()
        for label in (self.position_label, self.target_label, self.clearance_label):
            label.setStyleSheet("padding: 10px; background: white; border: 1px solid #d9e1e8;")
            readouts.addWidget(label, 1)
        layout.addLayout(readouts)
        self.message_label = QLabel("Ready for manual movement.")
        layout.addWidget(self.message_label)
        self.update_instrument()

        QApplication.instance().installEventFilter(self)
        self.last_tick = perf_counter()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_movement)
        self.timer.start(33)

    def build_scene(self):
        organ = pv.Sphere(radius=1, theta_resolution=64, phi_resolution=48)
        organ.points = organ.points * np.array([20, 26, 22]) + np.array([110, 0, 0])
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
        self.label_actor = self.viewport.add_point_labels(
            np.array([self.port, self.target, self.structure_centre, [110, 0, 22]]),
            ["Fixed port", "Surface target", "Protected structure", "Organ surface"],
            font_size=13, text_color="#243447", point_size=0, shape_opacity=0.8,
            always_visible=True)
        self.viewport.add_axes(color="#243447")
        self.viewport.add_legend([
            ["Instrument", "#526b80", "line"], ["Tool tip", "#d59420", "circle"],
            ["Target", "#199a78", "circle"], ["Protected structure", "#bc4046", "circle"],
        ], bcolor="white", border=False, size=(0.31, 0.13), loc="upper left", font_family="arial")
        self.reset_camera()

    def reset_camera(self):
        self.viewport.camera_position = [(170, -190, 145), (65, 0, 0), (0, 0, 1)]
        self.viewport.render()

    def set_command(self, axis):
        value = self.sliders[axis].value() / 10
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
        self.message_label.setText("Movement paused." if paused else "Ready for manual movement.")

    def reset_instrument(self):
        self.pressed_keys.clear()
        self.configuration = np.array([0.0, 0.0, 35.0])
        self.command = self.configuration.copy()
        self.pause_button.setChecked(False)
        self.sync_controls()
        self.update_instrument()
        self.message_label.setText("Instrument reset to its starting configuration.")

    def toggle_labels(self, visible):
        self.label_actor.SetVisibility(visible)
        self.viewport.render()

    def advance_movement(self):
        now = perf_counter()
        # limits delayed frames to prevent sudden movement jumps
        timestep = min(now - self.last_tick, 0.05)
        self.last_tick = now
        if self.pause_button.isChecked():
            return

        if self.pressed_keys:
            velocity = np.zeros(3)
            for key in self.pressed_keys:
                axis, direction = self.key_directions[key]
                velocity[axis] += direction * self.speed_limits[axis]
            self.command = np.clip(self.configuration + velocity * timestep,
                                   self.lower_limits, self.upper_limits)
            self.sync_controls()

        change = np.clip(self.command - self.configuration,
                         -self.speed_limits * timestep, self.speed_limits * timestep)
        if not np.any(change):
            return
        proposed = self.configuration + change
        if movement_is_clear(self.port, self.configuration, proposed, self.tool_radius,
                             self.structure_centre, self.structure_radius,
                             self.required_clearance):
            self.configuration = proposed
            self.update_instrument()
            self.message_label.setText("Manual movement · shaft clearance maintained.")
        else:
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
                        self.pressed_keys.add(event.key())
                    else:
                        self.pressed_keys.discard(event.key())
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        self.timer.stop()
        QApplication.instance().removeEventFilter(self)
        self.viewport.close()
        super().closeEvent(event)
