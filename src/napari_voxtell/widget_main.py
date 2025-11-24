import time
from typing import Optional

import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import QWidget

from napari_voxtell.widget_gui import VoxtellGUI


class ProcessingThread(QThread):
    """Thread for processing to avoid freezing the UI."""

    finished = Signal(np.ndarray)

    def __init__(self, image_data):
        super().__init__()
        self.image_data = image_data

    def run(self):
        """Run the processing in a separate thread."""
        # Sleep for 3 seconds to simulate processing time
        time.sleep(3)

        # Generate a random mask with the same shape as the image
        random_mask = np.random.randint(0, 2, size=self.image_data.shape, dtype=np.uint8)

        self.finished.emit(random_mask)


class VoxtellWidget(VoxtellGUI):
    """
    A simplified widget for text-promptable segmentation in Napari.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        """
        Initialize the VoxtellWidget.
        """
        super().__init__(viewer, parent)
        self.mask_counter = 0
        self.processing_thread = None
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self._update_spinner)
        self.spinner_index = 0
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _update_spinner(self):
        """Update the spinner animation."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner = self.spinner_frames[self.spinner_index]
        self.status_label.setText(f"{spinner} Processing...")

    def _start_processing(self):
        """Start the processing animation."""
        self.submit_button.setEnabled(False)
        self.text_input.setEnabled(False)
        self.spinner_index = 0
        self.status_label.setText(f"{self.spinner_frames[0]} Processing...")
        self.spinner_timer.start(100)  # Update every 100ms

    def _stop_processing(self):
        """Stop the processing animation."""
        self.spinner_timer.stop()
        self.status_label.setText("✓ Done!")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))  # Clear after 2 seconds
        self.submit_button.setEnabled(True)
        self.text_input.setEnabled(True)

    def on_submit(self):
        """
        Handle text submission. For now, displays the text as a popup
        and creates a random mask.
        """
        from napari.utils.notifications import show_info, show_warning

        text = self.text_input.toPlainText()

        if not text:
            return

        # Show the text in a popup
        show_info(f"Text prompt: {text}")

        # Get the currently selected image layer
        image_layer = self.selected_image_layer
        if image_layer is None:
            show_warning("Please select an image layer first")
            return

        # Get the image properties
        image_data = image_layer.data

        # Start processing animation
        self._start_processing()

        # Create and start the processing thread
        self.processing_thread = ProcessingThread(image_data)
        self.processing_thread.finished.connect(
            lambda mask: self._on_processing_finished(mask, image_layer)
        )
        self.processing_thread.start()

    def _on_processing_finished(self, random_mask, image_layer):
        """Handle the completion of processing."""
        # Stop processing animation
        self._stop_processing()

        # Create a new labels layer for each submission with unique name
        self.mask_counter += 1
        label_layer_name = f"Voxtell - Segmentation {self.mask_counter}"

        # Preserve all spatial properties from the image layer
        self._viewer.add_labels(
            random_mask,
            name=label_layer_name,
            scale=image_layer.scale,
            translate=image_layer.translate,
            rotate=image_layer.rotate,
            shear=image_layer.shear,
            affine=image_layer.affine,
            opacity=0.5,
        )

        # Clear the text input
        self.text_input.clear()
