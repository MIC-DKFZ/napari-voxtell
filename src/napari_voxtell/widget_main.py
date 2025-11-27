import time
from typing import Optional
import SimpleITK as sitk
import nibabel as nib

import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import QWidget
from napari.utils.notifications import show_info, show_warning, show_error
import torch

from napari_voxtell.widget_gui import VoxtellGUI

from nnunetv2.inference.VoxTellPredictor import VoxTellPredictor


class InitializationThread(QThread):
    """Thread for model initialization to avoid freezing the UI."""

    finished = Signal(object)  # Emits the predictor object
    error = Signal(str)  # Emits error message if initialization fails

    def __init__(self, model_dir, device):
        super().__init__()
        self.model_dir = model_dir
        self.device = device

    def run(self):
        """Initialize the model in a separate thread."""
        try:
            predictor = VoxTellPredictor(
                model_dir=self.model_dir,
                device=self.device
            )
            self.finished.emit(predictor)
        except Exception as e:
            self.error.emit(str(e))


class ProcessingThread(QThread):
    """Thread for processing to avoid freezing the UI."""

    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, predictor, image_data, text_prompts):
        super().__init__()
        self.predictor = predictor
        self.image_data = image_data
        self.text_prompts = text_prompts

    def run(self):
        """Run the processing in a separate thread."""
        try:
            voxtell_seg = self.predictor.predict_single_image(
                self.image_data, 
                self.text_prompts
            ).astype(np.uint8)[0]
            self.finished.emit(voxtell_seg)
        except Exception as e:
            self.error.emit(str(e))


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
        self.predictor = None  # Will be initialized when user clicks "Initialize Model"
        self.processing_thread = None
        self.initialization_thread = None
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self._update_spinner)
        self.spinner_index = 0
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        
        # Default model path
        self.default_model_path = "path"

    def _update_spinner(self):
        """Update the spinner animation."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner = self.spinner_frames[self.spinner_index]
        current_text = self.status_label.text()
        # Keep the message, just update the spinner
        if " " in current_text:
            message = current_text.split(" ", 1)[1]
            self.status_label.setText(f"{spinner} {message}")

    def _start_processing(self, message="Processing..."):
        """Start the processing animation."""
        self.submit_button.setEnabled(False)
        self.text_input.setEnabled(False)
        self.init_button.setEnabled(False)
        self.spinner_index = 0
        self.status_label.setText(f"{self.spinner_frames[0]} {message}")
        self.spinner_timer.start(100)  # Update every 100ms

    def _stop_processing(self, message="✓ Done!", restore_submit=True):
        """Stop the processing animation."""
        self.spinner_timer.stop()
        self.status_label.setText(message)
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))  # Clear after 2 seconds
        if restore_submit:
            self.submit_button.setEnabled(True)
            self.text_input.setEnabled(True)
        else:
            self._unlock_session()

    def on_init(self):
        """Initialize the VoxTell predictor with the selected model."""

        # Get model path from custom input or use default
        model_path = self.model_path_input.text().strip()
        if not model_path:
            model_path = self.default_model_path
            show_info(f"Using default model path")

        # Start initialization animation
        self._start_processing("Initializing model...")

        # Create device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        show_info(f"Initializing model on {device}...")

        # Create and start the initialization thread
        self.initialization_thread = InitializationThread(model_path, device)
        self.initialization_thread.finished.connect(self._on_initialization_finished)
        self.initialization_thread.error.connect(self._on_initialization_error)
        self.initialization_thread.start()

    def _on_initialization_finished(self, predictor):
        """Handle successful model initialization."""
        
        self.predictor = predictor
        self._stop_processing("✓ Model initialized!", restore_submit=True)
        self._lock_session()
        show_info("Model initialized successfully! You can now submit prompts.")

    def _on_initialization_error(self, error_message):
        """Handle model initialization error."""
        from napari.utils.notifications import show_error
        
        self._stop_processing("✗ Initialization failed!", restore_submit=False)
        show_error(f"Failed to initialize model: {error_message}")

    def on_submit(self):
        """
        Handle text submission and run segmentation.
        """

        # Check if model is initialized
        if self.predictor is None:
            show_warning("Please initialize the model first")
            return

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

        # Reorient the image to RAS using SimpleITK
        original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image_layer.metadata["direction"].flatten())
        itk_image = sitk.GetImageFromArray(image_data)
        itk_image.SetDirection(image_layer.metadata["direction"].flatten())
        itk_image = sitk.DICOMOrient(itk_image, "RAS")
        image_data = sitk.GetArrayFromImage(itk_image)

        # Start processing animation
        self._start_processing("Segmenting...")

        # Create and start the processing thread
        self.processing_thread = ProcessingThread(self.predictor, image_data, text)
        self.processing_thread.finished.connect(
            lambda mask: self._on_processing_finished(mask, image_layer, text, original_orientation)
        )
        self.processing_thread.error.connect(self._on_processing_error)
        self.processing_thread.start()

    def _on_processing_finished(self, mask, image_layer, text, original_orientation):
        """Handle the completion of processing."""
        # Stop processing animation
        self._stop_processing("✓ Segmentation complete!")

        # Reorient the mask back to the original orientation
        itk_mask = sitk.GetImageFromArray(mask)
        itk_mask.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))  # RAS direction
        itk_mask = sitk.DICOMOrient(itk_mask, original_orientation)
        mask = sitk.GetArrayFromImage(itk_mask)

        # Create a new labels layer for each submission with unique name
        self.mask_counter += 1
        label_layer_name = text[:100] + "..." if len(text) > 100 else text

        # Preserve all spatial properties from the image layer
        self._viewer.add_labels(
            mask,
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

    def _on_processing_error(self, error_message):
        """Handle segmentation processing error."""
        
        self._stop_processing("✗ Segmentation failed!")
        show_error(f"Segmentation failed: {error_message}")
