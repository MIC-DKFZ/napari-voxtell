from typing import Optional

from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QLabel,
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

    def on_submit(self):
        """Handle submit button click - to be implemented in subclass."""
        pass
