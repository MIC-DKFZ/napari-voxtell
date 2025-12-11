from typing import Optional

from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


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
        self.setMinimumWidth(self._width)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._viewer = viewer

        _main_layout = QVBoxLayout()
        self.setLayout(_main_layout)

        # Add model selection
        _main_layout.addWidget(self._init_model_selection())

        # Add initialization button
        _main_layout.addWidget(self._init_control_buttons())

        # Add image selection
        _main_layout.addWidget(self._init_image_selection())

        # Add text prompt input
        _main_layout.addWidget(self._init_text_prompt())

        # Add submit button
        _main_layout.addWidget(self._init_submit_button())

        # Add status label
        _main_layout.addWidget(self._init_status_label())

        # Add stretch to push everything to the top
        _main_layout.addStretch()

        # Initialize session state
        self._unlock_session()

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
        return _group_box

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
        """Initializes the text prompt input field."""
        _group_box = QGroupBox("Text Prompt:")
        _layout = QVBoxLayout()

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter segmentation prompt...")
        self.text_input.setMinimumHeight(80)  # Adjust height for multi-line text
        self.text_input.setMaximumHeight(150)  # Set a reasonable maximum height
        self.text_input.setAcceptRichText(False)  # Plain text only

        _layout.addWidget(self.text_input)
        _group_box.setLayout(_layout)
        return _group_box

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
        """Initializes the submit button."""
        _group_box = QGroupBox("")
        _layout = QVBoxLayout()

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)

        _layout.addWidget(self.submit_button)
        _group_box.setLayout(_layout)
        return _group_box

    def _init_status_label(self) -> QWidget:
        """Initializes the status label."""
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        return self.status_label

    def _unlock_session(self):
        """Unlock the session, allowing model and image selection."""
        self.init_button.setEnabled(True)
        self.submit_button.setEnabled(False)
        self.text_input.setEnabled(False)

    def _lock_session(self):
        """Lock the session after initialization, enabling segmentation."""
        self.init_button.setEnabled(False)
        self.submit_button.setEnabled(True)
        self.text_input.setEnabled(True)

    def on_model_selected(self):
        """Handle model selection change - to be implemented in subclass."""
        self._unlock_session()

    def on_init(self):
        """Handle initialization button click - to be implemented in subclass."""
        pass

    def on_submit(self):
        """Handle submit button click - to be implemented in subclass."""
        pass
