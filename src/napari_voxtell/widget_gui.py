import os
from typing import Optional

from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QCursor, QPainter, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_voxtell.presets import PRESETS

# Qt's "no maximum" sentinel; used to release the temporary width cap below.
_QWIDGETSIZE_MAX = 16777215


class _ResizeGrip(QWidget):
    """A thin horizontal handle; drag it up/down to resize ``target``'s height."""

    def __init__(self, target, minimum=60, maximum=1000, parent=None):
        super().__init__(parent)
        self._target = target
        self._minimum = minimum
        self._maximum = maximum
        self._start_y = None
        self._start_h = None
        self.setFixedHeight(10)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Drag to resize")

    def mousePressEvent(self, event):
        self._start_y = QCursor.pos().y()
        self._start_h = self._target.height()

    def mouseMoveEvent(self, event):
        if self._start_y is None:
            return
        delta = QCursor.pos().y() - self._start_y
        height = max(self._minimum, min(self._maximum, self._start_h + delta))
        self._target.setFixedHeight(height)

    def mouseReleaseEvent(self, event):
        self._start_y = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(140, 140, 140))
        cx, cy = self.width() // 2, self.height() // 2
        for dx in (-8, 0, 8):  # a few dots to hint the handle is draggable
            painter.drawPoint(cx + dx, cy - 1)
            painter.drawPoint(cx + dx, cy + 2)
        painter.end()


class VoxtellGUI(QWidget):
    """
    A simplified GUI for text-promptable segmentation.

    Args:
        viewer (Viewer): The Napari viewer instance to connect with the GUI.
        parent (Optional[QWidget], optional): The parent widget. Defaults to None.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._width = 300
        # Comfortable width the dock opens at; the user can drag it wider/narrower
        # once the layout has settled (see showEvent).
        self._open_width = 550
        # Reserve room for the scroll area's vertical scrollbar so content is not
        # clipped at the minimum width.
        self.setMinimumWidth(self._width + 18)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # napari does not propagate our size hint to the dock, so the dock would
        # otherwise open at Qt's 640 px default. Cap the width so the layout settles
        # at the intended size; showEvent re-applies this on every open and then
        # releases it so the user can still widen the dock.
        self.setMaximumWidth(self._open_width)
        self._viewer = viewer

        # Build all controls into a content widget...
        _content = QWidget()
        _main_layout = QVBoxLayout()
        _content.setLayout(_main_layout)

        # Add logo as a header at the top
        _main_layout.addWidget(self._init_logo())

        # Add model selection
        _main_layout.addWidget(self._init_model_selection())

        # Add local/remote inference-location toggle + connection settings
        _main_layout.addWidget(self._init_server_mode())

        # Add initialization button
        _main_layout.addWidget(self._init_control_buttons())

        # Add image selection
        _main_layout.addWidget(self._init_image_selection())

        # Add text prompt input
        _main_layout.addWidget(self._init_text_prompt())

        # Add submit button
        _main_layout.addWidget(self._init_submit_button())

        # Add export button
        _main_layout.addWidget(self._init_export_button())

        # Add colour legend (hidden until a segmentation is produced)
        _main_layout.addWidget(self._init_legend())

        # Add status label
        _main_layout.addWidget(self._init_status_label())

        # Add stretch to push everything to the top
        _main_layout.addStretch()

        # Apply the initial (Local) inference-mode layout now that every control exists.
        self._set_mode_ui(False)

        # ...then wrap it in a scroll area so resizing the prompt/legend boxes makes
        # the panel scroll instead of forcing the napari window to grow (which would
        # otherwise block resizing/maximising and push the bottom grip off-screen).
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(_content)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        _outer = QVBoxLayout()
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(self._scroll_area)
        self.setLayout(_outer)

        # Initialize session state
        self._unlock_session()

    def _init_logo(self) -> QLabel:
        """Loads the VoxTell logo image; rescaled to the column width on resize."""
        self._logo_label = QLabel()
        self._logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(os.path.dirname(__file__), "resources", "VoxTellPluginLogo.png")
        self._logo_pixmap = QPixmap(logo_path)
        self._rescale_logo()
        return self._logo_label

    def _rescale_logo(self):
        """Scale the logo to fill the current column width, preserving aspect ratio."""
        pixmap = getattr(self, "_logo_pixmap", None)
        if pixmap is None or pixmap.isNull():
            return
        # Reference the scroll viewport width: it tracks the real panel width and,
        # unlike the logo label's own width, does NOT depend on the pixmap size, so
        # scaling the logo can't feed back into the layout. Before the scroll area
        # exists (first call during __init__), fall back to the nominal open width.
        sa = getattr(self, "_scroll_area", None)
        avail = sa.viewport().width() if sa is not None else 0
        if avail < 50:
            avail = self._open_width
        width = max(50, avail - 20)  # fill the column, small margin for the scrollbar
        self._logo_label.setPixmap(pixmap.scaledToWidth(width, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        """Keep the logo fitted to the column as the dock is resized."""
        super().resizeEvent(event)
        self._rescale_logo()

    def showEvent(self, event):
        """Fit the logo and set the dock width each time the panel is shown."""
        super().showEvent(event)
        # Re-apply the width cap on every open (covers closing and reopening the
        # dock, which would otherwise reopen at napari's tiny default width).
        self.setMaximumWidth(self._open_width)
        # The resizeEvent during show can fire before the column width is final, so
        # rescale the logo again on the next event-loop pass with the settled width.
        QTimer.singleShot(0, self._rescale_logo)
        # Wait past the layout storm, then lift the cap so the dock keeps its
        # settled width but becomes freely resizable again.
        QTimer.singleShot(500, self._release_width_cap)

    def _release_width_cap(self):
        """Remove the initial width cap so the user can widen the dock."""
        self.setMaximumWidth(_QWIDGETSIZE_MAX)

    def _init_model_selection(self) -> QGroupBox:
        """Initializes the model selection combo box and path input."""
        _group_box = QGroupBox("Model Selection:")
        _layout = QVBoxLayout()

        # Model dropdown
        model_options = ["voxtell_v1.1", "voxtell_v1.0"]
        self.model_selection = QComboBox()
        self.model_selection.addItems(model_options)
        self.model_selection.currentIndexChanged.connect(self.on_model_selected)
        _layout.addWidget(self.model_selection)

        # Custom path input with clear button
        _path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Or paste model checkpoint path...")
        self.model_path_input.textChanged.connect(self.on_model_selected)
        _path_layout.addWidget(self.model_path_input)

        # Clear button
        self.clear_path_button = QPushButton("✕")
        self.clear_path_button.setFixedWidth(30)
        self.clear_path_button.clicked.connect(self._clear_model_path)
        _path_layout.addWidget(self.clear_path_button)

        _layout.addLayout(_path_layout)
        _group_box.setLayout(_layout)
        # Kept so remote mode can hide the (irrelevant) local model picker.
        self._model_group = _group_box
        return _group_box

    def _init_server_mode(self) -> QGroupBox:
        """Local/remote toggle plus the remote server URL, API key and upload button."""
        _group_box = QGroupBox("Inference Location:")
        _layout = QVBoxLayout()

        self.mode_selection = QComboBox()
        self.mode_selection.addItems(["Local (this machine)", "Remote server"])
        # Connect AFTER addItems so populating it does not fire the handler before the
        # rest of the widgets (init button, etc.) exist.
        self.mode_selection.currentIndexChanged.connect(self.on_mode_changed)
        _layout.addWidget(self.mode_selection)

        # Remote-only settings, shown only when "Remote server" is selected.
        self._remote_box = QWidget()
        _remote_layout = QVBoxLayout()
        _remote_layout.setContentsMargins(0, 0, 0, 0)

        _remote_layout.addWidget(QLabel("Server URL:"))
        self.server_url_input = QLineEdit()
        self.server_url_input.setPlaceholderText("http://127.0.0.1:1527")
        _remote_layout.addWidget(self.server_url_input)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("API key (optional)")
        _remote_layout.addWidget(self.api_key_input)

        _hint = QLabel(
            "Open a .nii/.nii.gz normally (drag-and-drop); it is uploaded to the "
            "server automatically on the first Submit."
        )
        _hint.setWordWrap(True)
        _hint.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        _remote_layout.addWidget(_hint)

        self._remote_box.setLayout(_remote_layout)
        _layout.addWidget(self._remote_box)

        _group_box.setLayout(_layout)
        return _group_box

    def _is_remote(self) -> bool:
        """True when the user has selected remote-server inference."""
        return self.mode_selection.currentIndex() == 1

    def _set_mode_ui(self, is_remote: bool):
        """Show/hide the remote settings and relabel the init button for the mode."""
        self._remote_box.setVisible(is_remote)
        self._model_group.setVisible(not is_remote)
        self.init_button.setText("Connect to server" if is_remote else "Initialize Model")

    def _clear_model_path(self):
        """Clear the model path input."""
        self.model_path_input.clear()
        self.on_model_selected()

    def _init_image_selection(self) -> QGroupBox:
        """Initializes the image selection combo box in a group box."""
        _group_box = QGroupBox("Image Selection:")
        _layout = QVBoxLayout()

        # Create a simple combo box for image layer selection
        self.image_selection = QComboBox()
        self.image_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_selection.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.image_selection.setMinimumContentsLength(8)

        # Populate with image layers
        self._update_image_layers()

        # Connect to layer events to update when layers change
        self._viewer.layers.events.inserted.connect(self._update_image_layers)
        self._viewer.layers.events.removed.connect(self._update_image_layers)

        _layout.addWidget(self.image_selection)
        _group_box.setLayout(_layout)
        return _group_box

    def _update_image_layers(self, event=None):
        """Update the image layer dropdown."""
        current_text = self.image_selection.currentText()
        self.image_selection.clear()

        # Add all Image layers
        image_layers = [layer for layer in self._viewer.layers if isinstance(layer, Image)]
        for layer in image_layers:
            self.image_selection.addItem(layer.name)

        # Try to restore previous selection
        index = self.image_selection.findText(current_text)
        if index >= 0:
            self.image_selection.setCurrentIndex(index)

    @property
    def selected_image_layer(self):
        """Get the currently selected image layer."""
        layer_name = self.image_selection.currentText()
        if layer_name and layer_name in self._viewer.layers:
            return self._viewer.layers[layer_name]
        return None

    def _init_text_prompt(self) -> QGroupBox:
        """Initializes the text prompt input field with preset shortcuts."""
        _group_box = QGroupBox("Text Prompts (one per line):")
        _layout = QVBoxLayout()

        # Preset dropdown + "Add" button: append a curated group of prompts.
        _preset_layout = QHBoxLayout()
        self.preset_selection = QComboBox()
        self.preset_selection.addItems(list(PRESETS))
        self.preset_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Allow the combo to shrink (long names elide here, full names show in the
        # dropdown) so it does not force a wide minimum panel width.
        self.preset_selection.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preset_selection.setMinimumContentsLength(8)
        _preset_layout.addWidget(self.preset_selection)
        self.add_preset_button = QPushButton("Add")
        self.add_preset_button.setToolTip("Add the selected preset's prompts to the text box")
        self.add_preset_button.clicked.connect(self._add_preset)
        _preset_layout.addWidget(self.add_preset_button)
        _layout.addLayout(_preset_layout)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "One prompt per line - each line becomes its own segmentation, e.g.\n"
            "liver\nspleen\nright kidney"
        )
        # Initial height; the grip below lets the user drag-resize it.
        self.text_input.setFixedHeight(180)
        self.text_input.setAcceptRichText(False)  # Plain text only
        self.text_input.textChanged.connect(self._update_prompt_count)
        _layout.addWidget(self.text_input)
        _layout.addWidget(_ResizeGrip(self.text_input, minimum=80))

        # Live feedback: makes it explicit that each line is a separate prompt.
        self.prompt_count_label = QLabel("0 prompts")
        self.prompt_count_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        _layout.addWidget(self.prompt_count_label)

        _group_box.setLayout(_layout)
        return _group_box

    def _count_prompts(self) -> int:
        """Number of non-empty prompt lines currently entered."""
        return len([ln for ln in self.text_input.toPlainText().splitlines() if ln.strip()])

    def _update_prompt_count(self):
        """Refresh the '<n> prompts' helper label below the text box."""
        n = self._count_prompts()
        self.prompt_count_label.setText(
            f"{n} prompt{'' if n == 1 else 's'} - one per line, "
            "numbered top to bottom (label 1, 2, 3, ...)"
        )

    def _add_preset(self):
        """Append the selected preset's prompts to the text box, skipping duplicates."""
        prompts = PRESETS.get(self.preset_selection.currentText(), [])
        existing = {
            line.strip().lower()
            for line in self.text_input.toPlainText().splitlines()
            if line.strip()
        }
        new = [p for p in prompts if p.lower() not in existing]
        if not new:
            return
        current = self.text_input.toPlainText().rstrip("\n")
        block = "\n".join(new)
        self.text_input.setPlainText(f"{current}\n{block}" if current else block)

    def _init_control_buttons(self) -> QGroupBox:
        """Initializes the initialize button."""
        _group_box = QGroupBox("")
        _layout = QVBoxLayout()

        self.init_button = QPushButton("Initialize Model")
        self.init_button.clicked.connect(self.on_init)

        _layout.addWidget(self.init_button)
        _group_box.setLayout(_layout)
        return _group_box

    def _init_submit_button(self) -> QGroupBox:
        """Initializes output name/mode options, submit/cancel buttons and progress bar."""
        _group_box = QGroupBox("")
        _layout = QVBoxLayout()

        # Optional output name: names the result layer and the default save filename.
        _name_row = QHBoxLayout()
        _name_row.addWidget(QLabel("Output name:"))
        self.output_name_input = QLineEdit()
        self.output_name_input.setPlaceholderText("optional - names the result layer & save file")
        _name_row.addWidget(self.output_name_input)
        _layout.addLayout(_name_row)

        # Output mode: one combined multi-label layer (default) vs one layer per prompt.
        self.separate_checkbox = QCheckBox("Separate layer per prompt")
        self.separate_checkbox.setToolTip(
            "Unchecked (default): all prompts go into ONE multi-label layer (label "
            "1, 2, 3, ...).\nChecked: each prompt becomes its own binary layer (lets you "
            "toggle/recolour them individually and keeps overlapping structures separate)."
        )
        _layout.addWidget(self.separate_checkbox)

        # Optional postprocessing: keep only the largest connected component per prompt.
        self.largest_cc_checkbox = QCheckBox("Keep largest only")
        self.largest_cc_checkbox.setToolTip(
            "Per prompt, keep only the single largest connected blob (removes scattered "
            "false positives).\nUse for single compact structures (liver, one rib); leave "
            "OFF for genuinely multi-part prompts (e.g. 'ribs', 'both kidneys')."
        )
        _layout.addWidget(self.largest_cc_checkbox)

        _btn_row = QHBoxLayout()
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)
        _btn_row.addWidget(self.submit_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        self.cancel_button.setEnabled(False)
        _btn_row.addWidget(self.cancel_button)
        _layout.addLayout(_btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        _layout.addWidget(self.progress_bar)

        _group_box.setLayout(_layout)
        return _group_box

    def _init_export_button(self) -> QGroupBox:
        """Initializes the multi-label export button."""
        _group_box = QGroupBox("")
        _layout = QVBoxLayout()

        self.export_button = QPushButton("Save segmentations as NIfTI")
        self.export_button.setToolTip(
            "Save ALL VoxTell segmentations of the selected image as one multi-label "
            "NIfTI in the image's original orientation, plus a JSON legend.\n"
            "Labels are numbered in prompt order: label 1 = first prompt line, 2 = "
            "second, ... (where masks overlap, a later prompt wins)."
        )
        self.export_button.clicked.connect(self.on_save_multilabel)

        _layout.addWidget(self.export_button)
        _group_box.setLayout(_layout)
        return _group_box

    def _init_legend(self) -> QGroupBox:
        """Initializes the colour -> prompt legend (scrollable, hidden until used)."""
        self._legend_box = QGroupBox("Legend (colour -> prompt):")
        _layout = QVBoxLayout()

        self.legend_label = QLabel("")
        self.legend_label.setTextFormat(Qt.RichText)
        self.legend_label.setWordWrap(True)
        self.legend_label.setAlignment(Qt.AlignTop)

        self._legend_scroll = QScrollArea()
        self._legend_scroll.setWidgetResizable(True)
        self._legend_scroll.setWidget(self.legend_label)
        self._legend_scroll.setFixedHeight(140)  # resizable via the grip below
        _layout.addWidget(self._legend_scroll)
        _layout.addWidget(_ResizeGrip(self._legend_scroll, minimum=60))

        self._legend_box.setLayout(_layout)
        self._legend_box.setVisible(False)
        return self._legend_box

    def update_legend(self, pairs):
        """Show a colour swatch + prompt for each ``(prompt, (r, g, b))`` pair."""
        if not pairs:
            self.legend_label.setText("")
            self._legend_box.setVisible(False)
            return
        rows = []
        for prompt, (r, g, b) in pairs:
            rgb = f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
            rows.append(f'<span style="color:{rgb};font-size:14px;">&#9632;</span>&nbsp;{prompt}')
        self.legend_label.setText("<br>".join(rows))
        self._legend_box.setVisible(True)

    def _init_status_label(self) -> QWidget:
        """Initializes the status label."""
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        return self.status_label

    def _set_prompting_enabled(self, enabled: bool):
        """Enable/disable the prompt controls (text box, presets, name, mode, submit)."""
        self.submit_button.setEnabled(enabled)
        self.text_input.setEnabled(enabled)
        self.preset_selection.setEnabled(enabled)
        self.add_preset_button.setEnabled(enabled)
        self.output_name_input.setEnabled(enabled)
        self.separate_checkbox.setEnabled(enabled)
        self.largest_cc_checkbox.setEnabled(enabled)

    def _unlock_session(self):
        """Unlock the session, allowing model and image selection."""
        self.init_button.setEnabled(True)
        self._set_prompting_enabled(False)

    def _lock_session(self):
        """Lock the session after initialization, enabling segmentation."""
        self.init_button.setEnabled(False)
        self._set_prompting_enabled(True)

    def on_model_selected(self):
        """Handle model selection change - to be implemented in subclass."""
        self._unlock_session()

    def on_init(self):
        """Handle initialization button click - to be implemented in subclass."""

    def on_submit(self):
        """Handle submit button click - to be implemented in subclass."""

    def on_save_multilabel(self):
        """Handle export button click - to be implemented in subclass."""

    def on_cancel(self):
        """Handle cancel button click - to be implemented in subclass."""

    def on_mode_changed(self):
        """Handle local/remote mode change - to be implemented in subclass."""
        self._set_mode_ui(self._is_remote())
