from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from vesselme.data.models import ImageItem
from vesselme.services.label_service import LabelService
from vesselme.services.project_service import ProjectService
from vesselme.ui.canvas_widget import CanvasWidget
from vesselme.ui.icons import delete_icon, eye_icon, lock_icon, rename_icon


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.current_language = "en"
        self._zh_translations = self._load_zh_translations()
        self.setWindowTitle("VesselMe - Fundus Vessel Annotation")
        self.resize(1500, 920)

        self.project_service = ProjectService()
        self.label_service = LabelService()

        self.images: list[ImageItem] = []
        self.current_image_index = -1
        self.current_label_name: str | None = None
        self.current_image_rgb: np.ndarray | None = None

        self.setProperty("space_pressed", False)
        self._build_ui()
        self._bind_shortcuts()

    def _load_zh_translations(self) -> dict[str, str]:
        path = Path(__file__).resolve().parents[1] / "data" / "i18n_zh_CN.json"
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _tr(self, key: str, default: str) -> str:
        if self.current_language == "zh":
            return self._zh_translations.get(key, default)
        return default

    def _build_ui(self) -> None:
        file_panel = QWidget()
        file_panel.setObjectName("panel")
        file_layout = QVBoxLayout(file_panel)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(8)

        file_header = QHBoxLayout()
        self.file_title = QLabel("Project Files")
        self.file_title.setProperty("sectionTitle", True)
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setObjectName("primary")
        self.btn_open_folder.clicked.connect(self.open_folder)
        file_header.addWidget(self.file_title)
        file_header.addStretch(1)
        file_header.addWidget(self.btn_open_folder)

        self.folder_label = QLabel("Folder: -")
        self.folder_label.setProperty("muted", True)
        self.file_summary_label = QLabel("0 images")
        self.file_summary_label.setProperty("muted", True)

        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(56, 56))
        self.file_list.currentRowChanged.connect(self._on_file_selected)

        file_layout.addLayout(file_header)
        file_layout.addWidget(self.folder_label)
        file_layout.addWidget(self.file_summary_label)
        file_layout.addWidget(self.file_list, 1)

        canvas_panel = QWidget()
        canvas_panel.setObjectName("panel")
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(6, 6, 6, 6)
        canvas_layout.setSpacing(6)

        canvas_title_row = QHBoxLayout()
        self.canvas_title = QLabel("Canvas")
        self.canvas_title.setProperty("sectionTitle", True)
        self.canvas_hint = QLabel("Wheel to zoom  |  Right click to erase  |  Space+Drag to pan")
        self.canvas_hint.setProperty("muted", True)
        canvas_title_row.addWidget(self.canvas_title)
        canvas_title_row.addStretch(1)
        canvas_title_row.addWidget(self.canvas_hint)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("separator")

        self.canvas = CanvasWidget()
        self.canvas.zoomChanged.connect(self._update_status)
        self.canvas.brushChanged.connect(self._on_brush_changed)
        self.canvas.dirtyChanged.connect(self._on_canvas_dirty)
        self.canvas.message.connect(self._update_status)

        canvas_layout.addLayout(canvas_title_row)
        canvas_layout.addWidget(separator)
        canvas_layout.addWidget(self.canvas, 1)

        label_panel = QWidget()
        label_panel.setObjectName("panel")
        label_layout = QVBoxLayout(label_panel)
        label_layout.setContentsMargins(10, 10, 10, 10)
        label_layout.setSpacing(8)

        self.label_title = QLabel("Labels")
        self.label_title.setProperty("sectionTitle", True)
        self.label_list = QListWidget()
        self.label_list.currentItemChanged.connect(self._on_label_selected_item)

        row1 = QHBoxLayout()
        self.btn_new_label = QPushButton("New")
        self.btn_new_label.clicked.connect(self.create_label)
        self.btn_import = QPushButton("Import label (.tar)")
        self.btn_import.clicked.connect(self.import_label)
        row1.addWidget(self.btn_new_label)
        row1.addWidget(self.btn_import)

        row4 = QHBoxLayout()
        self.btn_save = QPushButton("Save (Ctrl+S)")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self.save_current_label)
        self.btn_export = QPushButton("Export Stroke PNG")
        self.btn_export.clicked.connect(self.export_stroke)
        row4.addWidget(self.btn_save)
        row4.addWidget(self.btn_export)

        self.label_name_edit = QLineEdit()
        self.label_name_edit.setPlaceholderText("Selected label name")
        self.label_name_edit.returnPressed.connect(self.rename_label_from_input)

        brush_row = QHBoxLayout()
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(1, 100)  # 0.5px to 50.0px, step 0.5
        self.brush_size_slider.setSingleStep(1)
        self.brush_size_slider.setPageStep(2)
        self.brush_size_slider.setValue(int(round(self.canvas.brush_size * 2)))
        self.brush_size_slider.valueChanged.connect(self._on_brush_slider_changed)
        self.brush_size_spin = QDoubleSpinBox()
        self.brush_size_spin.setRange(0.5, 50.0)
        self.brush_size_spin.setSingleStep(0.5)
        self.brush_size_spin.setDecimals(1)
        self.brush_size_spin.setValue(self.canvas.brush_size)
        self.brush_size_spin.valueChanged.connect(self._on_brush_spin_changed)
        brush_row.addWidget(self.brush_size_slider, 1)
        brush_row.addWidget(self.brush_size_spin)

        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setSingleStep(1)
        self.opacity_slider.setValue(int(round(self.canvas.get_overlay_opacity())))
        self.opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        self.opacity_value_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_value_label.setProperty("muted", True)
        self.opacity_value_label.setFixedWidth(40)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_value_label)

        label_layout.addWidget(self.label_title)
        label_layout.addWidget(self.label_list, 1)
        label_layout.addLayout(row1)
        self.brush_size_title = QLabel("Brush size")
        label_layout.addWidget(self.brush_size_title)
        label_layout.addLayout(brush_row)
        self.opacity_title = QLabel("Opacity")
        label_layout.addWidget(self.opacity_title)
        label_layout.addLayout(opacity_row)
        label_layout.addLayout(row4)

        splitter = QSplitter()
        splitter.addWidget(file_panel)
        splitter.addWidget(canvas_panel)
        splitter.addWidget(label_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setHandleWidth(8)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 7)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([320, 980, 420])
        file_panel.setMinimumWidth(240)
        canvas_panel.setMinimumWidth(420)
        label_panel.setMinimumWidth(300)
        self.setCentralWidget(splitter)

        main_toolbar = QToolBar("Main")
        main_toolbar.setObjectName("mainToolbar")
        main_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, main_toolbar)

        self.toolbar_open_action = QAction("Open Folder", self)
        self.toolbar_open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.toolbar_open_action.setToolTip("Choose an image folder and load all supported files for annotation.")
        self.toolbar_open_action.triggered.connect(self.open_folder)
        main_toolbar.addAction(self.toolbar_open_action)
        main_toolbar.addSeparator()

        self.toolbar_save_action = QAction("Save", self)
        self.toolbar_save_action.setShortcut(QKeySequence("Ctrl+S"))
        self.toolbar_save_action.setToolTip("Save the currently selected label mask to its .tar file.")
        self.toolbar_save_action.triggered.connect(self.save_current_label)
        main_toolbar.addAction(self.toolbar_save_action)

        self.toolbar_import_action = QAction("Import label (.tar)", self)
        self.toolbar_import_action.setToolTip("Import a .tar label package into memory for the current image.")
        self.toolbar_import_action.triggered.connect(self.import_label)
        main_toolbar.addAction(self.toolbar_import_action)

        self.toolbar_import_image_action = QAction("Import from image", self)
        self.toolbar_import_image_action.setToolTip("Import a mask image, binarize it, and attach it to a label.")
        self.toolbar_import_image_action.triggered.connect(self.import_label_from_image)
        main_toolbar.addAction(self.toolbar_import_image_action)

        self.toolbar_export_action = QAction("Export PNG", self)
        self.toolbar_export_action.setToolTip("Export the current label as a black-background PNG preview.")
        self.toolbar_export_action.triggered.connect(self.export_stroke)
        main_toolbar.addAction(self.toolbar_export_action)
        main_toolbar.addSeparator()

        self.toolbar_prev_action = QAction("Prev", self)
        self.toolbar_prev_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self.toolbar_prev_action.setToolTip("Go to the previous image in the current folder.")
        self.toolbar_prev_action.triggered.connect(self.prev_image)
        main_toolbar.addAction(self.toolbar_prev_action)

        self.toolbar_next_action = QAction("Next", self)
        self.toolbar_next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self.toolbar_next_action.setToolTip("Go to the next image in the current folder.")
        self.toolbar_next_action.triggered.connect(self.next_image)
        main_toolbar.addAction(self.toolbar_next_action)

        toolbar_spacer = QWidget()
        toolbar_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_toolbar.addWidget(toolbar_spacer)

        self.lang_menu_button = QToolButton()
        self.lang_menu_button.setObjectName("langMenuButton")
        self.lang_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.lang_menu_button.setText("EN ▾")
        self.lang_menu_button.setToolTip("Switch UI language")
        self.lang_menu = QMenu(self.lang_menu_button)
        self.lang_action_en = self.lang_menu.addAction("English")
        self.lang_action_en.setCheckable(True)
        self.lang_action_en.triggered.connect(lambda: self.set_language("en"))
        self.lang_action_zh = self.lang_menu.addAction("中文")
        self.lang_action_zh.setCheckable(True)
        self.lang_action_zh.triggered.connect(lambda: self.set_language("zh"))
        self.lang_menu_button.setMenu(self.lang_menu)
        main_toolbar.addWidget(self.lang_menu_button)

        left_toolbar = QToolBar("Tools")
        left_toolbar.setObjectName("leftToolsToolbar")
        left_toolbar.setOrientation(Qt.Orientation.Vertical)
        left_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, left_toolbar)

        self.btn_brush_tool = QToolButton()
        self.btn_brush_tool.setText("✏️")
        self.btn_brush_tool.setToolTip("Brush (B)")
        self.btn_brush_tool.setCheckable(True)
        self.btn_brush_tool.clicked.connect(lambda: self.set_tool("brush"))
        left_toolbar.addWidget(self.btn_brush_tool)

        self.btn_eraser_tool = QToolButton()
        self.btn_eraser_tool.setText("")
        self.btn_eraser_tool.setIcon(QIcon(str(Path(__file__).resolve().parents[1] / "assert" / "eraser.png")))
        self.btn_eraser_tool.setIconSize(QSize(18, 18))
        self.btn_eraser_tool.setToolTip("Eraser (E)")
        self.btn_eraser_tool.setCheckable(True)
        self.btn_eraser_tool.clicked.connect(lambda: self.set_tool("eraser"))
        left_toolbar.addWidget(self.btn_eraser_tool)

        self.btn_clear = QToolButton()
        self.btn_clear.setText("🗑️")
        self.btn_clear.setToolTip("Clear Label")
        self.btn_clear.clicked.connect(self.clear_current_label)
        left_toolbar.addWidget(self.btn_clear)

        self.btn_help_tool = QToolButton()
        self.btn_help_tool.setText("?")
        self.btn_help_tool.setToolTip("Quick tutorial")
        self.btn_help_tool.clicked.connect(self.show_quick_tutorial)
        left_toolbar.addWidget(self.btn_help_tool)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._build_actions()
        self._set_tool_buttons()
        self._apply_theme()
        self._apply_language()
        self._update_status()

    def _set_tool_buttons(self) -> None:
        is_brush = self.canvas.current_tool == "brush"
        is_eraser = self.canvas.current_tool == "eraser"
        self.btn_brush_tool.setChecked(is_brush)
        self.btn_eraser_tool.setChecked(is_eraser)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f6f8;
                color: #1f2933;
            }
            QWidget#panel {
                background: #f8fafc;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
            }
            QToolBar#mainToolbar {
                spacing: 6px;
                padding: 6px;
                background: #ffffff;
                border-bottom: 1px solid #d9e2ec;
            }
            QToolBar {
                spacing: 5px;
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 4px;
            }
            QLabel[sectionTitle="true"] {
                font-size: 14px;
                font-weight: 600;
                color: #102a43;
            }
            QLabel[muted="true"] {
                color: #627d98;
                font-size: 12px;
            }
            QFrame#separator {
                color: #d9e2ec;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
                padding: 2px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background: #dceafe;
                color: #102a43;
            }
            QPushButton, QToolButton {
                background: #ffffff;
                border: 1px solid #bcccdc;
                border-radius: 6px;
                padding: 6px 10px;
                color: #243b53;
            }
            QPushButton:hover, QToolButton:hover {
                background: #f0f4f8;
            }
            QPushButton:pressed, QToolButton:pressed {
                background: #d9e2ec;
            }
            QPushButton#primary {
                background: #1473e6;
                color: white;
                border-color: #0e5cc0;
            }
            QPushButton#primary:hover {
                background: #0f65cd;
            }
            QToolButton:checked {
                background: #dceafe;
                border-color: #1473e6;
                color: #0f4c9a;
                font-weight: 600;
            }
            QToolButton#labelIconButton {
                background: transparent;
                border: none;
                border-radius: 11px;
                padding: 0;
                min-width: 22px;
                min-height: 22px;
                max-width: 22px;
                max-height: 22px;
            }
            QToolButton#labelIconButton:hover {
                background: #e6edf5;
            }
            QToolButton#labelIconButton:pressed {
                background: #d9e2ec;
            }
            QToolButton#labelSwatchButton {
                background: transparent;
                border: none;
                min-width: 14px;
                min-height: 14px;
                max-width: 14px;
                max-height: 14px;
                padding: 0;
            }
            QToolBar#leftToolsToolbar QToolButton {
                min-width: 34px;
                min-height: 34px;
                max-width: 34px;
                max-height: 34px;
                padding: 0;
                font-size: 16px;
            }
            QToolButton#langMenuButton {
                min-width: 56px;
                max-width: 64px;
                padding: 4px 8px;
                border-radius: 14px;
                border: 1px solid #9fb3c8;
                background: #f0f4f8;
                color: #334e68;
                font-size: 12px;
                font-weight: 700;
            }
            QToolButton#langMenuButton:hover {
                background: #e6edf5;
            }
            QToolButton#langMenuButton:pressed {
                background: #d9e2ec;
            }
            QToolButton#langMenuButton::menu-indicator {
                image: none;
                width: 0px;
            }
            QLineEdit, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #bcccdc;
                border-radius: 6px;
                padding: 6px;
            }
            QStatusBar {
                background: #102a43;
                color: #f0f4f8;
            }
            """
        )

    def _build_actions(self) -> None:
        menubar = self.menuBar()

        self.file_menu = menubar.addMenu("File")
        self.menu_open_action = QAction("Open Folder", self)
        self.menu_open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.menu_open_action.triggered.connect(self.open_folder)
        self.file_menu.addAction(self.menu_open_action)

        self.menu_save_action = QAction("Save Current Label", self)
        self.menu_save_action.setShortcut(QKeySequence("Ctrl+S"))
        self.menu_save_action.triggered.connect(self.save_current_label)
        self.file_menu.addAction(self.menu_save_action)

        self.menu_import_action = QAction("Import Label Tar", self)
        self.menu_import_action.triggered.connect(self.import_label)
        self.file_menu.addAction(self.menu_import_action)

        self.menu_import_image_action = QAction("Import Label from Image", self)
        self.menu_import_image_action.triggered.connect(self.import_label_from_image)
        self.file_menu.addAction(self.menu_import_image_action)

        self.menu_export_action = QAction("Export Stroke PNG", self)
        self.menu_export_action.triggered.connect(self.export_stroke)
        self.file_menu.addAction(self.menu_export_action)

        self.nav_menu = menubar.addMenu("Navigate")
        self.menu_prev_action = QAction("Previous Image", self)
        self.menu_prev_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self.menu_prev_action.triggered.connect(self.prev_image)
        self.nav_menu.addAction(self.menu_prev_action)

        self.menu_next_action = QAction("Next Image", self)
        self.menu_next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self.menu_next_action.triggered.connect(self.next_image)
        self.nav_menu.addAction(self.menu_next_action)

        self.view_menu = menubar.addMenu("View")
        self.menu_fit_action = QAction("Fit to Window", self)
        self.menu_fit_action.triggered.connect(self.canvas.fit_to_window)
        self.view_menu.addAction(self.menu_fit_action)

        self.menu_actual_size_action = QAction("Actual Size (1:1)", self)
        self.menu_actual_size_action.triggered.connect(self.canvas.actual_size)
        self.view_menu.addAction(self.menu_actual_size_action)

        self.edit_menu = menubar.addMenu("Edit")
        self.menu_undo_action = QAction("Undo", self)
        self.menu_undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.menu_undo_action.triggered.connect(self.undo)
        self.edit_menu.addAction(self.menu_undo_action)

        self.menu_redo_action = QAction("Redo", self)
        self.menu_redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        self.menu_redo_action.triggered.connect(self.redo)
        self.edit_menu.addAction(self.menu_redo_action)

        self.menu_brush_inc_action = QAction("Brush +", self)
        self.menu_brush_inc_action.setShortcut(QKeySequence("]"))
        self.menu_brush_inc_action.triggered.connect(lambda: self._set_brush(self.canvas.brush_size + 1))
        self.edit_menu.addAction(self.menu_brush_inc_action)

        self.menu_brush_dec_action = QAction("Brush -", self)
        self.menu_brush_dec_action.setShortcut(QKeySequence("["))
        self.menu_brush_dec_action.triggered.connect(lambda: self._set_brush(self.canvas.brush_size - 1))
        self.edit_menu.addAction(self.menu_brush_dec_action)

        self.menu_save_s_action = QAction("Save (S)", self)
        self.menu_save_s_action.setShortcut(QKeySequence("S"))
        self.menu_save_s_action.triggered.connect(self.save_current_label)
        self.edit_menu.addAction(self.menu_save_s_action)

        self.menu_toggle_overlay_action = QAction("Toggle Overlay", self)
        self.menu_toggle_overlay_action.setShortcut(QKeySequence("A"))
        self.menu_toggle_overlay_action.triggered.connect(self.toggle_overlay)
        self.edit_menu.addAction(self.menu_toggle_overlay_action)

    def set_language(self, language: str) -> None:
        if language not in {"en", "zh"}:
            return
        if language == self.current_language:
            return
        self.current_language = language
        self._apply_language()
        self._refresh_labels()
        if self.current_image_index >= 0:
            self._refresh_file_row(self.current_image_index)
        self._update_status()

    def _apply_language(self) -> None:
        self.setWindowTitle(self._tr("window.title", "VesselMe - Fundus Vessel Annotation"))
        self.file_title.setText(self._tr("panel.project_files", "Project Files"))
        self.canvas_title.setText(self._tr("panel.canvas", "Canvas"))
        self.label_title.setText(self._tr("panel.labels", "Labels"))
        self.btn_open_folder.setText(self._tr("btn.open_folder", "Open Folder"))
        self.btn_new_label.setText(self._tr("btn.new", "New"))
        self.btn_import.setText(self._tr("btn.import_label_tar", "Import label (.tar)"))
        self.btn_save.setText(self._tr("btn.save_ctrl_s", "Save (Ctrl+S)"))
        self.btn_export.setText(self._tr("btn.export_stroke_png", "Export Stroke PNG"))
        self.label_name_edit.setPlaceholderText(self._tr("placeholder.selected_label_name", "Selected label name"))
        self.brush_size_title.setText(self._tr("label.brush_size", "Brush size"))
        self.opacity_title.setText(self._tr("label.opacity", "Opacity"))
        self.canvas_hint.setText(
            self._tr(
                "hint.canvas_ops",
                "Wheel to zoom  |  Right click to erase  |  Space+Drag to pan",
            )
        )
        self.canvas.set_empty_hint_text(
            self._tr(
                "hint.canvas_empty",
                "Open Folder to start annotation\n\nB: Brush  E: Eraser  A: Toggle Overlay",
            )
        )

        self.toolbar_open_action.setText(self._tr("toolbar.open_folder", "Open Folder"))
        self.toolbar_open_action.setToolTip(
            self._tr("tooltip.toolbar_open_folder", "Choose an image folder and load all supported files for annotation.")
        )
        self.toolbar_save_action.setText(self._tr("toolbar.save", "Save"))
        self.toolbar_save_action.setToolTip(
            self._tr("tooltip.toolbar_save", "Save the currently selected label mask to its .tar file.")
        )
        self.toolbar_import_action.setText(self._tr("toolbar.import_label_tar", "Import label (.tar)"))
        self.toolbar_import_action.setToolTip(
            self._tr("tooltip.toolbar_import_tar", "Import a .tar label package into memory for the current image.")
        )
        self.toolbar_import_image_action.setText(self._tr("toolbar.import_from_image", "Import from image"))
        self.toolbar_import_image_action.setToolTip(
            self._tr("tooltip.toolbar_import_image", "Import a mask image, binarize it, and attach it to a label.")
        )
        self.toolbar_export_action.setText(self._tr("toolbar.export_png", "Export PNG"))
        self.toolbar_export_action.setToolTip(
            self._tr("tooltip.toolbar_export_png", "Export the current label as a black-background PNG preview.")
        )
        self.toolbar_prev_action.setText(self._tr("toolbar.prev", "Prev"))
        self.toolbar_prev_action.setToolTip(
            self._tr("tooltip.toolbar_prev", "Go to the previous image in the current folder.")
        )
        self.toolbar_next_action.setText(self._tr("toolbar.next", "Next"))
        self.toolbar_next_action.setToolTip(
            self._tr("tooltip.toolbar_next", "Go to the next image in the current folder.")
        )

        self.btn_brush_tool.setToolTip(self._tr("tooltip.brush", "Brush (B)"))
        self.btn_eraser_tool.setToolTip(self._tr("tooltip.eraser", "Eraser (E)"))
        self.btn_clear.setToolTip(self._tr("tooltip.clear_label", "Clear Label"))
        self.btn_help_tool.setToolTip(self._tr("tooltip.quick_tutorial", "Quick tutorial"))

        self.lang_menu_button.setText("EN ▾" if self.current_language == "en" else "中 ▾")
        self.lang_menu_button.setToolTip(self._tr("tooltip.switch_language", "Switch UI language"))
        self.lang_action_en.setText(self._tr("language.english", "English"))
        self.lang_action_zh.setText(self._tr("language.chinese", "中文"))
        self.lang_action_en.setChecked(self.current_language == "en")
        self.lang_action_zh.setChecked(self.current_language == "zh")

        self.file_menu.setTitle(self._tr("menu.file", "File"))
        self.nav_menu.setTitle(self._tr("menu.navigate", "Navigate"))
        self.view_menu.setTitle(self._tr("menu.view", "View"))
        self.edit_menu.setTitle(self._tr("menu.edit", "Edit"))
        self.menu_open_action.setText(self._tr("menu.open_folder", "Open Folder"))
        self.menu_save_action.setText(self._tr("menu.save_current_label", "Save Current Label"))
        self.menu_import_action.setText(self._tr("menu.import_label_tar", "Import Label Tar"))
        self.menu_import_image_action.setText(self._tr("menu.import_label_from_image", "Import Label from Image"))
        self.menu_export_action.setText(self._tr("menu.export_stroke_png", "Export Stroke PNG"))
        self.menu_prev_action.setText(self._tr("menu.previous_image", "Previous Image"))
        self.menu_next_action.setText(self._tr("menu.next_image", "Next Image"))
        self.menu_fit_action.setText(self._tr("menu.fit_to_window", "Fit to Window"))
        self.menu_actual_size_action.setText(self._tr("menu.actual_size", "Actual Size (1:1)"))
        self.menu_undo_action.setText(self._tr("menu.undo", "Undo"))
        self.menu_redo_action.setText(self._tr("menu.redo", "Redo"))
        self.menu_brush_inc_action.setText(self._tr("menu.brush_plus", "Brush +"))
        self.menu_brush_dec_action.setText(self._tr("menu.brush_minus", "Brush -"))
        self.menu_save_s_action.setText(self._tr("menu.save_s", "Save (S)"))
        self.menu_toggle_overlay_action.setText(self._tr("menu.toggle_overlay", "Toggle Overlay"))

        root_display = str(self.project_service.root_dir) if self.project_service.root_dir is not None else "-"
        self.folder_label.setText(f"{self._tr('label.folder', 'Folder')}: {root_display}")
        self.file_summary_label.setText(self._tr("label.images_count", "{count} images").format(count=len(self.images)))

    def show_quick_tutorial(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._tr("tutorial.window_title", "VesselMe Quick Guide"))
        dialog.setMinimumSize(760, 560)
        dialog.setStyleSheet(
            """
            QDialog {
                background: #f8fafc;
            }
            QLabel#guideTitle {
                font-size: 22px;
                font-weight: 700;
                color: #102a43;
            }
            QLabel#guideSubtitle {
                font-size: 13px;
                color: #486581;
            }
            QLabel#sectionTitle {
                font-size: 14px;
                font-weight: 700;
                color: #243b53;
                padding-top: 6px;
            }
            QLabel#bodyText {
                font-size: 13px;
                color: #243b53;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                gridline-color: #e4e7eb;
                font-size: 13px;
                color: #102a43;
            }
            QHeaderView::section {
                background: #eef2f7;
                color: #243b53;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #d9e2ec;
                font-weight: 600;
            }
            """
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.setSpacing(10)

        title = QLabel(self._tr("tutorial.title", "VesselMe Annotation Guide"))
        title.setObjectName("guideTitle")

        subtitle = QLabel(
            self._tr(
                "tutorial.subtitle",
                "VesselMe is a local annotation tool for fundus vessel segmentation. "
                "Each label is edited in memory and saved as a separate .tar package.",
            )
        )
        subtitle.setObjectName("guideSubtitle")
        subtitle.setWordWrap(True)

        basics_title = QLabel(self._tr("tutorial.basic_workflow", "Basic Workflow"))
        basics_title.setObjectName("sectionTitle")
        basics = QLabel(
            "\n".join(
                [
                    self._tr("tutorial.step1", "1. Open your image folder."),
                    self._tr("tutorial.step2", "2. Pick an image from the file list."),
                    self._tr("tutorial.step3", "3. Create a label or import one."),
                    self._tr("tutorial.step4", "4. Paint vessels with Brush/Eraser."),
                    self._tr("tutorial.step5", "5. Save the current label to <image>_[<label>].tar."),
                ]
            )
        )
        basics.setObjectName("bodyText")
        basics.setWordWrap(True)

        shortcuts_title = QLabel(self._tr("tutorial.shortcut_menu", "Shortcut Menu"))
        shortcuts_title.setObjectName("sectionTitle")

        shortcut_rows = [
            (self._tr("tutorial.action.brush", "Brush"), "B"),
            (self._tr("tutorial.action.eraser", "Eraser"), "E"),
            (self._tr("tutorial.action.adjust_brush", "Adjust brush size"), "Ctrl + Mouse Wheel"),
            (self._tr("tutorial.action.temp_eraser", "Temporary eraser"), self._tr("tutorial.shortcut.right_click", "Right Click")),
            (self._tr("tutorial.action.pan_canvas", "Pan canvas"), "Space + Left Drag / Middle Drag"),
            (self._tr("tutorial.action.zoom", "Zoom"), "Mouse Wheel"),
            (self._tr("tutorial.action.undo_redo", "Undo / Redo"), "Ctrl+Z / Ctrl+Y"),
            (self._tr("tutorial.action.toggle_overlay", "Toggle overlay"), "A"),
            (self._tr("tutorial.action.brush_size_pm", "Brush size - / +"), "[ / ]"),
            (self._tr("tutorial.action.switch_label", "Switch label"), "1 ~ 9"),
            (self._tr("tutorial.action.save_current_label", "Save current label"), "S or Ctrl+S"),
            (self._tr("tutorial.action.prev_next_image", "Previous / Next image"), "Left / Right Arrow"),
        ]

        table = QTableWidget(len(shortcut_rows), 2, dialog)
        table.setHorizontalHeaderLabels(
            [
                self._tr("tutorial.header.action", "Action"),
                self._tr("tutorial.header.shortcut", "Shortcut"),
            ]
        )
        table.verticalHeader().setVisible(False)
        table.setShowGrid(True)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        for row, (action_name, shortcut) in enumerate(shortcut_rows):
            table.setItem(row, 0, QTableWidgetItem(action_name))
            table.setItem(row, 1, QTableWidgetItem(shortcut))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, parent=dialog)
        buttons.accepted.connect(dialog.accept)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(basics_title)
        layout.addWidget(basics)
        layout.addWidget(shortcuts_title)
        layout.addWidget(table, 1)
        layout.addWidget(buttons)

        dialog.exec()

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("B"), self, activated=lambda: self.set_tool("brush"))
        QShortcut(QKeySequence("E"), self, activated=lambda: self.set_tool("eraser"))
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("A"), self, activated=self.toggle_overlay)
        QShortcut(QKeySequence("S"), self, activated=self.save_current_label)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current_label)

        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, activated=lambda idx=i: self.select_label_by_index(idx - 1))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.setProperty("space_pressed", True)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.setProperty("space_pressed", False)
        super().keyReleaseEvent(event)

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self._tr("dialog.select_image_folder", "Select Image Folder"))
        if not folder:
            return

        path = Path(folder)
        try:
            self.images = self.project_service.open_folder(path)
        except Exception as exc:
            QMessageBox.critical(self, self._tr("error.open_folder_failed", "Open folder failed"), str(exc))
            return

        self.folder_label.setText(f"{self._tr('label.folder', 'Folder')}: {path}")
        self._populate_file_list()

        if self.images:
            self.file_list.setCurrentRow(0)
        else:
            self.current_image_index = -1
            self.current_label_name = None
            self.label_list.clear()
            self.current_image_rgb = None
            self.canvas.update()
            self._update_status(self._tr("status.no_images_found", "No images found"))
        self.file_summary_label.setText(self._tr("label.images_count", "{count} images").format(count=len(self.images)))

    def _display_rel_path(self, item: ImageItem) -> str:
        rel_path = item.path.name
        if self.project_service.root_dir is not None:
            try:
                rel_path = str(item.path.relative_to(self.project_service.root_dir))
            except ValueError:
                rel_path = item.path.name
        return rel_path

    def _build_file_row_widget(self, item: ImageItem) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(6, 4, 6, 4)
        row_layout.setSpacing(8)

        thumb_label = QLabel()
        thumb_label.setFixedSize(56, 56)
        icon = self._make_thumbnail_icon(item.path)
        if icon is not None:
            pm = icon.pixmap(56, 56)
            thumb_label.setPixmap(pm)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(4)

        name_label = QLabel(self._display_rel_path(item))
        if item.has_dirty_label:
            name_label.setStyleSheet("color: rgba(188, 45, 45, 180);")
        else:
            name_label.setStyleSheet("color: #102a43;")

        dots_row = QHBoxLayout()
        dots_row.setContentsMargins(0, 0, 0, 0)
        dots_row.setSpacing(4)
        for label_name in sorted(item.labels.keys()):
            label = item.labels[label_name]
            dot = QWidget()
            dot.setFixedSize(8, 8)
            dot.setToolTip(label_name)
            dot.setStyleSheet(
                "background: rgb(%d, %d, %d); border-radius: 4px;"
                % (label.display_color[0], label.display_color[1], label.display_color[2])
            )
            dots_row.addWidget(dot)
        dots_row.addStretch(1)

        text_col.addWidget(name_label)
        text_col.addLayout(dots_row)

        row_layout.addWidget(thumb_label)
        row_layout.addLayout(text_col, 1)
        return row_widget

    def _populate_file_list(self) -> None:
        self.file_list.clear()
        for item in self.images:
            row = QListWidgetItem()
            row.setSizeHint(QSize(0, 66))
            self.file_list.addItem(row)
            self.file_list.setItemWidget(row, self._build_file_row_widget(item))
        self.file_summary_label.setText(self._tr("label.images_count", "{count} images").format(count=len(self.images)))

    def _make_thumbnail_icon(self, image_path: Path) -> QIcon | None:
        qimg = QImage(str(image_path))
        if qimg.isNull():
            return None
        pix = QPixmap.fromImage(qimg).scaled(
            56,
            56,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(pix)

    def _refresh_file_row(self, index: int) -> None:
        if index < 0 or index >= len(self.images):
            return
        row_item = self.file_list.item(index)
        if row_item is None:
            return
        row_item.setSizeHint(QSize(0, 66))
        self.file_list.setItemWidget(row_item, self._build_file_row_widget(self.images[index]))

    def _on_file_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.images):
            return
        if row == self.current_image_index:
            return

        if not self._guard_unsaved_before_switch():
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_image_index)
            self.file_list.blockSignals(False)
            return

        self.current_image_index = row
        item = self.images[row]
        try:
            self.current_image_rgb = self.project_service.load_image_rgb(item.path)
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.image_load_failed", "Image load failed"), str(exc))
            return

        self._refresh_labels()
        if self.label_list.count() > 0:
            self.label_list.setCurrentRow(0)
        else:
            self.current_label_name = None
            self.label_name_edit.clear()
            self.canvas.set_image_preview(self.current_image_rgb)
            self.canvas.set_editable(False)
        self._refresh_file_row(row)
        self._update_status()

    def _refresh_labels(self) -> None:
        self.label_list.clear()
        item = self.current_image_item
        if item is None:
            return

        for idx, name in enumerate(sorted(item.labels.keys())):
            label = item.labels[name]
            dirty = "*" if label.dirty else ""

            row = QListWidgetItem()
            row.setData(Qt.ItemDataRole.UserRole, name)
            row.setSizeHint(QSize(0, 38))
            self.label_list.addItem(row)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 3, 6, 3)
            row_layout.setSpacing(6)

            swatch = QToolButton()
            swatch.setFixedSize(14, 14)
            swatch.setObjectName("labelSwatchButton")
            swatch.setToolTip(
                self._tr("tooltip.change_color_of_label", "Change color of {name}").format(name=name)
            )
            swatch.setStyleSheet(
                "QToolButton {"
                f"background: rgb({label.display_color[0]}, {label.display_color[1]}, {label.display_color[2]});"
                "border: 1px solid #334e68;"
                "border-radius: 7px;"
                "}"
            )
            swatch.clicked.connect(lambda checked=False, label_name=name: self.pick_label_color_by_name(label_name))

            text = QLabel(f"{idx+1}. {name}{dirty}")
            text.setStyleSheet("background: transparent;")

            eye_btn = QToolButton()
            eye_btn.setFixedSize(22, 22)
            eye_btn.setObjectName("labelIconButton")
            eye_btn.setIcon(eye_icon(label.visible, size=18))
            eye_btn.setIconSize(QSize(16, 16))
            eye_btn.setToolTip(
                self._tr("tooltip.hide_label", "Hide label")
                if label.visible
                else self._tr("tooltip.show_label", "Show label")
            )
            eye_btn.clicked.connect(lambda checked=False, label_name=name: self.toggle_label_visible_by_name(label_name))

            lock_btn = QToolButton()
            lock_btn.setFixedSize(22, 22)
            lock_btn.setObjectName("labelIconButton")
            lock_btn.setIcon(lock_icon(label.locked, size=18))
            lock_btn.setIconSize(QSize(16, 16))
            lock_btn.setToolTip(
                self._tr("tooltip.lock_label", "Lock label")
                if not label.locked
                else self._tr("tooltip.unlock_label", "Unlock label")
            )
            lock_btn.clicked.connect(lambda checked=False, label_name=name: self.toggle_label_lock_by_name(label_name))

            rename_btn = QToolButton()
            rename_btn.setFixedSize(22, 22)
            rename_btn.setObjectName("labelIconButton")
            rename_btn.setIcon(rename_icon(size=18))
            rename_btn.setIconSize(QSize(16, 16))
            rename_btn.setToolTip(self._tr("tooltip.rename_label", "Rename label"))
            rename_btn.clicked.connect(lambda checked=False, label_name=name: self.rename_label_by_name(label_name))

            delete_btn = QToolButton()
            delete_btn.setFixedSize(22, 22)
            delete_btn.setObjectName("labelIconButton")
            delete_btn.setIcon(delete_icon(size=18))
            delete_btn.setIconSize(QSize(16, 16))
            delete_btn.setToolTip(self._tr("tooltip.delete_label", "Delete label"))
            delete_btn.clicked.connect(lambda checked=False, label_name=name: self.delete_label_by_name(label_name))

            row_layout.addWidget(swatch)
            row_layout.addWidget(text, 1)
            row_layout.addWidget(eye_btn)
            row_layout.addWidget(lock_btn)
            row_layout.addWidget(rename_btn)
            row_layout.addWidget(delete_btn)
            self.label_list.setItemWidget(row, row_widget)

    @property
    def current_image_item(self) -> ImageItem | None:
        if self.current_image_index < 0 or self.current_image_index >= len(self.images):
            return None
        return self.images[self.current_image_index]

    @property
    def current_label(self):
        item = self.current_image_item
        if item is None or self.current_label_name is None:
            return None
        return item.labels.get(self.current_label_name)

    def _on_label_selected_item(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        item = self.current_image_item
        if item is None or self.current_image_rgb is None or current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(name, str):
            return
        self.current_label_name = name

        label = item.labels[name]
        label.ensure_mask(self.current_image_rgb.shape[:2])
        self.canvas.set_scene(self.current_image_rgb, label.mask, label.display_color, preserve_view=True)
        self.canvas.set_overlay_visible(label.visible)
        self.canvas.set_editable(not label.locked)
        self.label_name_edit.setText(name)
        self._update_status()

    def create_label(self) -> None:
        item = self.current_image_item
        if item is None or self.current_image_rgb is None:
            return

        default_name = self.label_service.make_default_name(item)
        name, ok = self._get_text_dialog(
            self._tr("dialog.create_label", "Create Label"),
            self._tr("dialog.label_name", "Label Name"),
            default_name,
        )
        if not ok:
            return

        try:
            self.label_service.create_label(item, name, self.current_image_rgb.shape[:2], color=(255, 255, 255))
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.create_label_failed", "Create label failed"), str(exc))
            return

        self._refresh_labels()
        self.select_label_by_name(name)
        self._refresh_file_row(self.current_image_index)

    def import_label(self) -> None:
        item = self.current_image_item
        if item is None or self.current_image_rgb is None:
            return

        tar_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("dialog.import_label_tar", "Import label (.tar)"),
            str(item.path.parent),
            self._tr("filter.tar_files", "Tar Files (*.tar)"),
        )
        if not tar_path:
            return

        try:
            label = self.label_service.import_tar(item, Path(tar_path), self.current_image_rgb.shape[:2])
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.import_label_failed", "Import label failed"), str(exc))
            return

        self._refresh_labels()
        self.select_label_by_name(label.label_name)
        self._refresh_file_row(self.current_image_index)

    def import_label_from_image(self) -> None:
        item = self.current_image_item
        if item is None or self.current_image_rgb is None:
            return

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle(self._tr("dialog.import_from_image", "Import from image"))
        msg.setText(self._tr("dialog.import_from_image_question", "How do you want to import this mask image?"))
        msg.setInformativeText(
            self._tr(
                "dialog.import_from_image_info",
                "Choose to create a new label or overwrite the current selected label.",
            )
        )
        new_btn = msg.addButton(self._tr("btn.create_new_label", "Create new label"), QMessageBox.ButtonRole.AcceptRole)
        overwrite_btn = msg.addButton(
            self._tr("btn.overwrite_current_label", "Overwrite current label"),
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_btn = msg.addButton(self._tr("btn.cancel", "Cancel"), QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == cancel_btn or clicked is None:
            return
        mode = "new" if clicked == new_btn else "overwrite"

        if mode == "overwrite" and self.current_label is None:
            QMessageBox.warning(
                self,
                self._tr("dialog.import_from_image", "Import from image"),
                self._tr(
                    "warn.no_label_selected_for_overwrite",
                    "No label is selected. Select a label first or choose create new label.",
                ),
            )
            return

        label_name: str | None = None
        label_color = (255, 255, 255)
        if mode == "new":
            default_name = self.label_service.make_default_name(item)
            customize = QMessageBox(self)
            customize.setIcon(QMessageBox.Icon.Question)
            customize.setWindowTitle(self._tr("dialog.new_label_options", "New label options"))
            customize.setText(self._tr("dialog.customize_label_before_import", "Customize label name and color before importing?"))
            yes_btn = customize.addButton(self._tr("btn.customize", "Customize"), QMessageBox.ButtonRole.AcceptRole)
            no_btn = customize.addButton(self._tr("btn.use_defaults", "Use defaults"), QMessageBox.ButtonRole.NoRole)
            cancel_customize_btn = customize.addButton(self._tr("btn.cancel", "Cancel"), QMessageBox.ButtonRole.RejectRole)
            customize.exec()
            chosen = customize.clickedButton()
            if chosen == cancel_customize_btn or chosen is None:
                return
            if chosen == yes_btn:
                custom_name, ok = self._get_text_dialog(
                    self._tr("dialog.create_label_from_image", "Create Label from Image"),
                    self._tr("dialog.label_name", "Label Name"),
                    default_name,
                )
                if not ok:
                    return
                label_name = custom_name

                color = QColorDialog.getColor(
                    QColor(*label_color),
                    self,
                    self._tr("dialog.pick_label_color_optional", "Pick Label Color (Optional)"),
                )
                if color.isValid():
                    label_color = (color.red(), color.green(), color.blue())
            elif chosen == no_btn:
                label_name = default_name

        image_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("dialog.import_mask_image", "Import Mask Image"),
            str(item.path.parent),
            self._tr("filter.image_files", "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"),
        )
        if not image_path:
            return

        try:
            if mode == "new":
                label = self.label_service.import_image_as_new_label(
                    item,
                    Path(image_path),
                    self.current_image_rgb.shape[:2],
                    label_name=label_name,
                    color=label_color,
                )
            else:
                assert self.current_label_name is not None
                label = self.label_service.overwrite_label_mask_from_image(
                    item,
                    self.current_label_name,
                    Path(image_path),
                    self.current_image_rgb.shape[:2],
                )
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.import_from_image_failed", "Import from image failed"), str(exc))
            return

        self._refresh_labels()
        self.select_label_by_name(label.label_name)
        label.ensure_mask(self.current_image_rgb.shape[:2])
        self.canvas.set_scene(self.current_image_rgb, label.mask, label.display_color, preserve_view=True)
        self.canvas.set_overlay_visible(label.visible)
        self.canvas.set_editable(not label.locked)
        self._refresh_file_row(self.current_image_index)
        self._update_status(
            self._tr(
                "status.imported_mask_memory_only",
                "Imported mask image into {label_name} (memory only, save to write .tar)",
            ).format(label_name=label.label_name)
        )

    def rename_label(self) -> None:
        label = self.current_label
        item = self.current_image_item
        if label is None or item is None:
            return

        new_name, ok = self._get_text_dialog(
            self._tr("dialog.rename_label", "Rename Label"),
            self._tr("dialog.new_name", "New Name"),
            label.label_name,
        )
        if not ok:
            return

        old_name = label.label_name
        try:
            self.label_service.rename_label(item, old_name, new_name)
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.rename_failed", "Rename failed"), str(exc))
            return

        self.current_label_name = new_name
        self._refresh_labels()
        self.select_label_by_name(new_name)
        self._refresh_file_row(self.current_image_index)

    def rename_label_by_name(self, label_name: str) -> None:
        item = self.current_image_item
        if item is None or label_name not in item.labels:
            return
        self.current_label_name = label_name
        self.select_label_by_name(label_name)
        self.rename_label()

    def rename_label_from_input(self) -> None:
        label = self.current_label
        if label is None:
            return
        new_name = self.label_name_edit.text().strip()
        if new_name and new_name != label.label_name:
            self.rename_label_with_name(new_name)

    def rename_label_with_name(self, new_name: str) -> None:
        label = self.current_label
        item = self.current_image_item
        if label is None or item is None:
            return

        old = label.label_name
        try:
            self.label_service.rename_label(item, old, new_name)
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.rename_failed", "Rename failed"), str(exc))
            self.label_name_edit.setText(old)
            return

        self.current_label_name = new_name
        self._refresh_labels()
        self.select_label_by_name(new_name)
        self._refresh_file_row(self.current_image_index)

    def delete_label(self) -> None:
        label = self.current_label
        item = self.current_image_item
        if label is None or item is None:
            return

        ans = QMessageBox.question(
            self,
            self._tr("dialog.delete_label", "Delete label"),
            self._tr("dialog.delete_label_confirm", "Delete label '{label_name}' and its .tar file?").format(
                label_name=label.label_name
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        self.label_service.delete_label(item, label.label_name)
        self.current_label_name = None
        self._refresh_labels()
        if self.label_list.count() > 0:
            self.label_list.setCurrentRow(0)
        else:
            self.label_name_edit.clear()
            if self.current_image_rgb is not None:
                self.canvas.set_image_preview(self.current_image_rgb)
                self.canvas.set_editable(False)
            else:
                self.canvas.update()
        self._refresh_file_row(self.current_image_index)

    def delete_label_by_name(self, label_name: str) -> None:
        item = self.current_image_item
        if item is None or label_name not in item.labels:
            return
        self.current_label_name = label_name
        self.select_label_by_name(label_name)
        self.delete_label()

    def pick_label_color(self) -> None:
        label = self.current_label
        if label is None:
            return
        self.pick_label_color_by_name(label.label_name)

    def pick_label_color_by_name(self, label_name: str) -> None:
        item = self.current_image_item
        if item is None or label_name not in item.labels:
            return
        label = item.labels[label_name]

        color = QColorDialog.getColor(
            QColor(*label.display_color),
            self,
            self._tr("dialog.pick_label_color", "Pick Label Color"),
        )
        if not color.isValid():
            return

        label.display_color = (color.red(), color.green(), color.blue())
        label.dirty = True
        if self.current_label_name == label_name:
            self.canvas.set_overlay_color(label.display_color)
        self._refresh_labels()
        self.select_label_by_name(label_name)

    def toggle_overlay(self) -> None:
        label = self.current_label
        if label is None:
            return
        label.visible = not label.visible
        label.dirty = True
        self.canvas.set_overlay_visible(label.visible)
        self._refresh_labels()
        self._refresh_file_row(self.current_image_index)

    def toggle_label_visible_by_name(self, label_name: str) -> None:
        item = self.current_image_item
        if item is None or label_name not in item.labels:
            return
        label = item.labels[label_name]
        label.visible = not label.visible
        label.dirty = True
        if self.current_label_name == label_name:
            self.canvas.set_overlay_visible(label.visible)
        self._refresh_labels()
        self.select_label_by_name(label_name)
        self._refresh_file_row(self.current_image_index)

    def toggle_lock(self) -> None:
        label = self.current_label
        if label is None:
            return
        label.locked = not label.locked
        label.dirty = True
        self.canvas.set_editable(not label.locked)
        self._refresh_labels()
        self._refresh_file_row(self.current_image_index)
        state = self._tr("status.locked", "locked") if label.locked else self._tr("status.unlocked", "unlocked")
        self._update_status(self._tr("status.label_is_state", "Label {label_name} is {state}").format(label_name=label.label_name, state=state))

    def toggle_label_lock_by_name(self, label_name: str) -> None:
        item = self.current_image_item
        if item is None or label_name not in item.labels:
            return
        label = item.labels[label_name]
        label.locked = not label.locked
        label.dirty = True
        if self.current_label_name == label_name:
            self.canvas.set_editable(not label.locked)
            state = self._tr("status.locked", "locked") if label.locked else self._tr("status.unlocked", "unlocked")
            self._update_status(
                self._tr("status.label_is_state", "Label {label_name} is {state}").format(
                    label_name=label.label_name,
                    state=state,
                )
            )
        self._refresh_labels()
        self.select_label_by_name(label_name)
        self._refresh_file_row(self.current_image_index)

    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self._set_tool_buttons()
        self._update_status()

    def _set_brush(self, size: float) -> None:
        value = max(0.5, min(50.0, float(size)))
        self.canvas.set_brush_size(value)
        self.brush_size_slider.blockSignals(True)
        self.brush_size_slider.setValue(int(round(self.canvas.brush_size * 2)))
        self.brush_size_slider.blockSignals(False)
        self.brush_size_spin.blockSignals(True)
        self.brush_size_spin.setValue(self.canvas.brush_size)
        self.brush_size_spin.blockSignals(False)
        self._update_status()

    def _on_brush_slider_changed(self, value: int) -> None:
        self._set_brush(value / 2.0)

    def _on_brush_spin_changed(self, value: float) -> None:
        self._set_brush(value)

    def _on_brush_changed(self, _: float) -> None:
        self.brush_size_slider.blockSignals(True)
        self.brush_size_slider.setValue(int(round(self.canvas.brush_size * 2)))
        self.brush_size_slider.blockSignals(False)
        self.brush_size_spin.blockSignals(True)
        self.brush_size_spin.setValue(self.canvas.brush_size)
        self.brush_size_spin.blockSignals(False)
        self._update_status()

    def _on_opacity_slider_changed(self, value: int) -> None:
        self.canvas.set_overlay_opacity(float(value))
        self.opacity_value_label.setText(f"{value}%")
        self._update_status()

    def _on_canvas_dirty(self) -> None:
        label = self.current_label
        if label is None:
            return
        label.dirty = True
        self._refresh_labels()
        self._refresh_file_row(self.current_image_index)
        self._update_status()

    def undo(self) -> None:
        if self.current_label and self.current_label.locked:
            self._update_status(self._tr("status.current_label_locked", "Current label is locked."))
            return
        if self.canvas.undo():
            self._on_canvas_dirty()

    def redo(self) -> None:
        if self.current_label and self.current_label.locked:
            self._update_status(self._tr("status.current_label_locked", "Current label is locked."))
            return
        if self.canvas.redo():
            self._on_canvas_dirty()

    def clear_current_label(self) -> None:
        if self.current_label and self.current_label.locked:
            self._update_status(self._tr("status.current_label_locked", "Current label is locked."))
            return
        if self.canvas.clear_mask():
            self._on_canvas_dirty()

    def save_current_label(self) -> None:
        item = self.current_image_item
        label = self.current_label
        if item is None or label is None:
            return

        try:
            path = self.label_service.save_label(item, label.label_name)
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.save_failed", "Save failed"), str(exc))
            return

        self._refresh_labels()
        self._refresh_file_row(self.current_image_index)
        self._update_status(self._tr("status.saved_file", "Saved {name}").format(name=path.name))

    def export_stroke(self) -> None:
        item = self.current_image_item
        label = self.current_label
        if item is None or label is None:
            return

        default_name = f"{item.stem}_[{label.label_name}]_stroke.png"
        default_path = str(item.path.parent / default_name)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("dialog.export_stroke_png", "Export Stroke PNG"),
            default_path,
            self._tr("filter.png_files", "PNG Files (*.png)"),
        )
        if not save_path:
            return

        try:
            path = self.label_service.export_stroke(item, label.label_name, output_path=Path(save_path))
        except Exception as exc:
            QMessageBox.warning(self, self._tr("error.export_failed", "Export failed"), str(exc))
            return

        self._update_status(self._tr("status.exported_path", "Exported {path}").format(path=path))

    def _guard_unsaved_before_switch(self) -> bool:
        item = self.current_image_item
        if item is None:
            return True

        dirty_labels = [lbl for lbl in item.labels.values() if lbl.dirty]
        if not dirty_labels:
            return True

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(self._tr("dialog.unsaved_labels", "Unsaved labels"))
        msg.setText(self._tr("dialog.unsaved_edits", "Current image has unsaved label edits."))
        msg.setInformativeText(self._tr("dialog.save_discard_cancel", "Save / Discard / Cancel"))
        save_btn = msg.addButton(self._tr("btn.save", "Save"), QMessageBox.ButtonRole.AcceptRole)
        discard_btn = msg.addButton(self._tr("btn.discard", "Discard"), QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton(self._tr("btn.cancel", "Cancel"), QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == cancel_btn:
            return False
        if clicked == save_btn:
            for label in dirty_labels:
                try:
                    self.label_service.save_label(item, label.label_name)
                except Exception as exc:
                    QMessageBox.warning(self, self._tr("error.save_failed", "Save failed"), str(exc))
                    return False
        if clicked == discard_btn:
            for label in dirty_labels:
                label.dirty = False
        return True

    def prev_image(self) -> None:
        if not self.images:
            return
        row = max(0, self.current_image_index - 1)
        self.file_list.setCurrentRow(row)

    def next_image(self) -> None:
        if not self.images:
            return
        row = min(len(self.images) - 1, self.current_image_index + 1)
        self.file_list.setCurrentRow(row)

    def select_label_by_index(self, index: int) -> None:
        if index < 0 or index >= self.label_list.count():
            return
        self.label_list.setCurrentRow(index)

    def select_label_by_name(self, name: str) -> None:
        for i in range(self.label_list.count()):
            row_item = self.label_list.item(i)
            if row_item.data(Qt.ItemDataRole.UserRole) == name:
                self.label_list.setCurrentRow(i)
                return

    def closeEvent(self, event) -> None:
        if self._guard_unsaved_before_switch():
            event.accept()
        else:
            event.ignore()

    def _get_text_dialog(self, title: str, label: str, value: str) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(self, title, label, text=value)
        return text.strip(), ok and bool(text.strip())

    def _update_status(self, message: str | None = None) -> None:
        item = self.current_image_item
        label = self.current_label

        fname = item.name if item else "-"
        zoom = f"{self.canvas.scale * 100:.0f}%"
        brush = f"{self.canvas.brush_size:.1f}px"
        opacity = f"{self.canvas.get_overlay_opacity():.0f}%"
        lname = label.label_name if label else "-"
        save_state = self._tr("status.dirty", "dirty") if label and label.dirty else self._tr("status.saved", "saved")
        if label is None:
            lock_state = self._tr("status.no_label", "no-label")
        else:
            lock_state = self._tr("status.locked", "locked") if label.locked else self._tr("status.editable", "editable")
        tool = self.canvas.current_tool
        tool_text = (
            self._tr("tool.brush", "brush")
            if tool == "brush"
            else self._tr("tool.eraser", "eraser")
            if tool == "eraser"
            else tool
        )
        base = (
            self._tr(
                "status.base",
                "File: {fname} | Zoom: {zoom} | Brush: {brush} | Tool: {tool} | Label: {lname} | Opacity: {opacity} | {lock_state} | {save_state}",
            ).format(
                fname=fname,
                zoom=zoom,
                brush=brush,
                tool=tool_text,
                lname=lname,
                opacity=opacity,
                lock_state=lock_state,
                save_state=save_state,
            )
        )

        if message:
            self.status.showMessage(f"{base} | {message}")
        else:
            self.status.showMessage(base)
