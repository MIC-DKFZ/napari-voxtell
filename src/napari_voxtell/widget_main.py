import contextlib
import gc
import inspect
import json
import os
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_error, show_info, show_warning
from napari.viewer import Viewer
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import QFileDialog, QWidget
from voxtell.inference.predictor import VoxTellPredictor

try:  # older voxtell releases lack cooperative cancellation
    from voxtell.inference.predictor import InferenceCancelled
except ImportError:  # pragma: no cover

    class InferenceCancelled(RuntimeError):
        pass


from napari_voxtell.widget_gui import VoxtellGUI

# Whether the installed voxtell supports the progress/cancel hook.
_PREDICT_SUPPORTS_PROGRESS = (
    "progress_callback" in inspect.signature(VoxTellPredictor.predict_single_image).parameters
)

# Qualitative palette cycled across generated segmentation layers so each prompt
# gets a visually distinct colour (RGB in 0-1).
_LABEL_COLORS = [
    (0.90, 0.10, 0.10),
    (0.12, 0.47, 0.71),
    (0.20, 0.63, 0.17),
    (1.00, 0.50, 0.00),
    (0.42, 0.24, 0.60),
    (0.65, 0.34, 0.16),
    (0.89, 0.47, 0.76),
    (0.30, 0.69, 0.29),
    (0.99, 0.75, 0.44),
    (0.55, 0.63, 0.80),
]


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary mask to its single largest connected component."""
    from scipy.ndimage import label as cc_label

    labeled, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return (labeled == sizes.argmax()).astype(mask.dtype)


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
            predictor = VoxTellPredictor(model_dir=self.model_dir, device=self.device)
            self.finished.emit(predictor)
        except Exception as e:
            self.error.emit(str(e))


class ProcessingThread(QThread):
    """Thread for processing to avoid freezing the UI."""

    finished = Signal(np.ndarray)
    error = Signal(str)
    progress = Signal(int, int)  # (patches done, total patches)
    cancelled = Signal()
    oom_fallback = Signal(str)  # notify the UI we are retrying after a GPU-OOM

    def __init__(self, predictor, image_data, text_prompts, keep_largest=False):
        super().__init__()
        self.predictor = predictor
        self.image_data = image_data
        self.text_prompts = text_prompts
        self.keep_largest = keep_largest
        self._cancel = False

    def cancel(self):
        """Request cooperative cancellation (checked after each sliding-window patch)."""
        self._cancel = True

    def _progress_cb(self, done, total):
        self.progress.emit(done, total)
        return not self._cancel

    def _predict(self, prompts, on_device):
        """Run one prediction with the logits accumulator on GPU (fast) or CPU (safe).

        One predictor call embeds all prompts and runs the image encoder once per
        patch, looping over prompts only in the lightweight decoder. Shape:
        (num_prompts, Z, Y, X).
        """
        self.predictor.perform_everything_on_device = on_device
        kwargs = {"progress_callback": self._progress_cb} if _PREDICT_SUPPORTS_PROGRESS else {}
        return self.predictor.predict_single_image(self.image_data, prompts, **kwargs).astype(
            np.uint8
        )

    @staticmethod
    def _is_oom(err) -> bool:
        return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()

    def _free_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _initial_batch_size(self, n_prompts: int) -> int:
        """Largest prompt batch likely to fit in the currently free GPU memory.

        The per-prompt logits accumulator (~volume x 2 bytes) dominates, so we size the
        batch from free VRAM up-front instead of trying all prompts and OOM-ing slowly.
        """
        device = getattr(self.predictor, "device", None)
        if not (torch.cuda.is_available() and getattr(device, "type", "") == "cuda"):
            return n_prompts
        try:
            free, _ = torch.cuda.mem_get_info()
        except Exception:  # noqa: BLE001 - any failure -> just try all prompts at once
            return n_prompts
        voxels = int(np.prod(self.image_data.shape[-3:]))
        per_prompt = voxels * 2  # half-precision logits accumulator, bytes
        # Reserve ~60% of free memory for the network, activations and n_predictions.
        fit = int(free * 0.4 / max(per_prompt, 1))
        return max(1, min(n_prompts, fit))

    def _predict_with_oom_fallbacks(self) -> np.ndarray:
        """Predict in GPU batches sized to fit memory, degrading gracefully on OOM.

        Batch size starts from free VRAM; on an unexpected OOM it halves, and a lone
        prompt that still will not fit falls back to (slow but safe) CPU accumulation.
        """
        original = getattr(self.predictor, "perform_everything_on_device", True)
        try:
            prompts = list(self.text_prompts)
            size = self._initial_batch_size(len(prompts))
            if size < len(prompts):
                self.oom_fallback.emit(
                    f"Segmenting in batches of {size} prompt(s) to fit GPU memory..."
                )
            masks, i = [], 0
            while i < len(prompts):
                sub = prompts[i : i + size]
                try:
                    masks.append(self._predict(sub, on_device=True))
                    i += size
                except InferenceCancelled:
                    raise
                except RuntimeError as e:
                    if not self._is_oom(e):
                        raise
                    self._free_gpu()
                    if size > 1:
                        size = max(1, size // 2)
                        self.oom_fallback.emit(f"Low GPU memory - reducing batch to {size}...")
                        continue
                    self.oom_fallback.emit(
                        "A prompt is too large for GPU memory - using CPU accumulation (slow)..."
                    )
                    masks.append(self._predict(sub, on_device=False))
                    i += 1
            return masks[0] if len(masks) == 1 else np.concatenate(masks, axis=0)
        finally:
            self.predictor.perform_everything_on_device = original

    def run(self):
        """Run the processing in a separate thread."""
        try:
            voxtell_seg = self._predict_with_oom_fallbacks()
            if self.keep_largest:
                voxtell_seg = np.stack([_keep_largest_component(m) for m in voxtell_seg])
            self.finished.emit(voxtell_seg)
        except InferenceCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Release cached GPU blocks so memory does not creep up across runs.
            self._free_gpu()


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

        # Keep the legend in sync with the selected segmentation layer (and its colours).
        self._legend_layer = None
        self._viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

    def _on_active_layer_changed(self, event=None):
        """Show the legend of the active VoxTell layer; track its colour changes."""
        # Stop listening to the previously tracked layer's colormap.
        if self._legend_layer is not None:
            with contextlib.suppress(TypeError, ValueError, RuntimeError):
                self._legend_layer.events.colormap.disconnect(self._refresh_legend)
            self._legend_layer = None

        layer = self._viewer.layers.selection.active
        if isinstance(layer, Labels) and layer.metadata.get("voxtell_legend"):
            self._legend_layer = layer
            layer.events.colormap.connect(self._refresh_legend)
            self._refresh_legend()
        else:
            self.update_legend([])

    def _refresh_legend(self, event=None):
        """Rebuild the legend from the tracked layer's CURRENT colours (survives recolor).

        Uses ``colormap.map(value)`` rather than ``color_dict`` because napari's
        "shuffle/recolor" replaces the DirectLabelColormap with a CyclicLabelColormap
        that has no ``color_dict`` - reading it would blank the legend.
        """
        layer = self._legend_layer
        if layer is None:
            return
        legend = layer.metadata.get("voxtell_legend", {})
        pairs = []
        for value, name in legend.items():
            pairs.append((name, self._label_rgb(layer, value)))
        self.update_legend(pairs)

    @staticmethod
    def _label_rgb(layer, value):
        """Current displayed RGB (0-1) for a label value, for any colormap type."""
        try:
            rgba = np.asarray(layer.colormap.map(np.array([value]))).reshape(-1, 4)[0]
            return tuple(float(c) for c in rgba[:3])
        except Exception:  # noqa: BLE001 - fall back to a neutral grey on any colormap quirk
            return (0.5, 0.5, 0.5)

    def _update_spinner(self):
        """Update the spinner animation."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner = self.spinner_frames[self.spinner_index]
        current_text = self.status_label.text()
        # Keep the message, just update the spinner
        if " " in current_text:
            message = current_text.split(" ", 1)[1]
            self.status_label.setText(f"{spinner} {message}")

    def _start_processing(self, message="Processing...", cancellable=False):
        """Start the processing animation (and progress bar / cancel for inference)."""
        self.init_button.setEnabled(False)
        self._set_prompting_enabled(False)
        self.spinner_index = 0
        self.status_label.setText(f"{self.spinner_frames[0]} {message}")
        self.spinner_timer.start(100)  # Update every 100ms
        if cancellable:
            self.cancel_button.setEnabled(True)
            self.progress_bar.setRange(0, 0)  # busy until the first patch reports in
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

    def _stop_processing(self, message="✓ Done!", restore_submit=True):
        """Stop the processing animation."""
        self.spinner_timer.stop()
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))  # Clear after 2 seconds
        if restore_submit:
            self._set_prompting_enabled(True)
        else:
            self._unlock_session()

    def on_init(self):
        """Initialize the VoxTell predictor with the selected model."""

        # Get model path from custom input or use selected model
        model_path = self.model_path_input.text().strip()
        if not model_path:
            # Use the selected model from dropdown
            selected_model = self.model_selection.currentText()

            if selected_model not in ["voxtell_v1.0", "voxtell_v1.1"]:
                show_error(f"Unknown model selected: {selected_model}")
                return
            if selected_model == "voxtell_v1.0":
                show_warning(
                    "VoxTell v1.0 is deprecated. Please use v1.1 for better performance and features."
                )

            repo_id = "mrokuss/VoxTell"
            dowload_path = snapshot_download(
                repo_id=repo_id, allow_patterns=[f"{selected_model}/*", "*.json"]
            )

            model_path = os.path.join(dowload_path, selected_model)
            if not os.path.exists(model_path):
                show_error(f"Could not fetch {selected_model}")
                return
        else:
            show_info(f"Using custom model path: {model_path}")

        # Start initialization animation
        self._start_processing("Initializing model...")

        # Create device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model on {device}...")

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

        # One prompt per non-empty line; each yields its own segmentation layer.
        prompts = [
            line.strip() for line in self.text_input.toPlainText().splitlines() if line.strip()
        ]

        if not prompts:
            return

        show_info(f"Segmenting {len(prompts)} prompt(s): {', '.join(prompts)}")

        # Get the currently selected image layer
        image_layer = self.selected_image_layer
        if image_layer is None:
            show_warning("Please select an image layer first")
            return

        # VoxTell expects images in its training orientation (RAS, via nnU-Net's
        # NibabelIOWithReorient). The VoxTell reader loads them that way and tags the
        # layer; other readers leave the volume in a different orientation, which the
        # model has never seen and which yields mis-aligned, low-quality masks.
        if not image_layer.metadata.get("voxtell_reoriented"):
            show_warning(
                "This image was not opened with the VoxTell reader, so its orientation "
                "may not match the model. Please reopen the image and choose the "
                "'napari-voxtell' reader."
            )
            return

        # Already in the model's input space - feed it straight to the predictor.
        image_data = np.asarray(image_layer.data)

        separate = self.separate_checkbox.isChecked()

        # Start processing animation
        self._start_processing("Segmenting...", cancellable=True)

        # Create and start the processing thread
        self.processing_thread = ProcessingThread(
            self.predictor,
            image_data,
            prompts,
            keep_largest=self.largest_cc_checkbox.isChecked(),
        )
        self.processing_thread.finished.connect(
            lambda masks: self._on_processing_finished(masks, image_layer, prompts, separate)
        )
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.cancelled.connect(self._on_processing_cancelled)
        self.processing_thread.oom_fallback.connect(self._on_oom_fallback)
        self.processing_thread.error.connect(self._on_processing_error)
        self.processing_thread.start()

    def _on_processing_finished(self, masks, image_layer, prompts, separate):
        """Handle the completion of processing.

        ``masks`` has shape (num_prompts, Z, Y, X), in the displayed image's
        (reoriented) space so it overlays directly. Depending on ``separate`` the
        result is shown either as one combined multi-label layer or as one binary
        layer per prompt.
        """
        # Stop processing animation
        self._stop_processing("✓ Segmentation complete!")

        empty = [p for p, m in zip(prompts, masks) if not m.any()]
        if separate:
            self._add_separate_layers(masks, image_layer, prompts)
        else:
            self._add_multilabel_layer(masks, image_layer, prompts)

        # Reflect the newly added (now active) layer in the legend.
        self._on_active_layer_changed()

        if empty:
            show_warning("No voxels found for: " + ", ".join(empty))

        # Clear the text input
        self.text_input.clear()

    def _spatial_kwargs(self, image_layer) -> dict:
        """Spatial layer kwargs so a result overlays the source image exactly."""
        return {
            "scale": image_layer.scale,
            "translate": image_layer.translate,
            "rotate": image_layer.rotate,
            "shear": image_layer.shear,
            "affine": image_layer.affine,
            "opacity": 0.5,
        }

    def _add_multilabel_layer(self, masks, image_layer, prompts):
        """Add one combined multi-label layer (prompt i -> value i+1)."""
        combined = np.zeros(masks.shape[1:], dtype=np.uint16)
        legend = {}
        color_dict = {None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0)}
        for value, (prompt, mask) in enumerate(zip(prompts, masks), start=1):
            combined[mask > 0] = value
            legend[value] = prompt
            color_dict[value] = (*_LABEL_COLORS[(value - 1) % len(_LABEL_COLORS)], 1.0)

        self.mask_counter += 1
        name = self.output_name_input.text().strip() or self._segmentation_layer_name(prompts)
        self._viewer.add_labels(
            combined,
            name=name,
            colormap=DirectLabelColormap(color_dict=color_dict),
            metadata={
                "voxtell_props": image_layer.metadata.get("voxtell_props"),
                "voxtell_source": image_layer.name,
                "voxtell_legend": legend,
            },
            **self._spatial_kwargs(image_layer),
        )
        # Single layer hides names; report the value -> prompt mapping (legend follows
        # the layer selection, see _on_active_layer_changed).
        show_info("Labels: " + ", ".join(f"{v}={p}" for v, p in legend.items()))

    def _add_separate_layers(self, masks, image_layer, prompts):
        """Add one binary Labels layer per prompt, each a distinct colour."""
        for prompt, mask in zip(prompts, masks):
            color = _LABEL_COLORS[self.mask_counter % len(_LABEL_COLORS)]
            self.mask_counter += 1
            name = prompt[:100] + "..." if len(prompt) > 100 else prompt
            self._viewer.add_labels(
                mask.astype(np.uint8),
                name=name,
                colormap=DirectLabelColormap(
                    color_dict={
                        None: (0.0, 0.0, 0.0, 0.0),
                        0: (0.0, 0.0, 0.0, 0.0),
                        1: (*color, 1.0),
                    }
                ),
                metadata={
                    "voxtell_props": image_layer.metadata.get("voxtell_props"),
                    "voxtell_source": image_layer.name,
                    "voxtell_legend": {1: prompt},
                },
                **self._spatial_kwargs(image_layer),
            )

    def _segmentation_layer_name(self, prompts) -> str:
        """A compact layer name summarising the prompts (napari de-duplicates if needed)."""
        head = ", ".join(prompts[:3])
        if len(prompts) > 3:
            head += f", +{len(prompts) - 3}"
        return head

    def _on_progress(self, done, total):
        """Update the progress bar as sliding-window patches complete."""
        if self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def on_cancel(self):
        """Request cancellation of the running segmentation."""
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling...")
            self.processing_thread.cancel()

    def _on_processing_cancelled(self):
        """Handle a user-cancelled segmentation."""
        self._stop_processing("Segmentation cancelled.")
        show_info("Segmentation cancelled.")

    def _on_oom_fallback(self, message):
        """Notify the user the segmentation is retrying after a GPU out-of-memory."""
        show_warning(message)
        # The retry restarts the sliding window, so reset the progress bar to busy.
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(f"{self.spinner_frames[self.spinner_index]} {message}")

    def _on_processing_error(self, error_message):
        """Handle segmentation processing error."""

        self._stop_processing("✗ Segmentation failed!")
        show_error(f"Segmentation failed: {error_message}")

    def on_save_multilabel(self):
        """Save all VoxTell segmentations of the selected image as one multi-label NIfTI.

        Every VoxTell Labels layer belonging to the selected image (whether one
        combined multi-label layer or several per-prompt layers) is merged into a
        single labelmap: each prompt gets a fresh label value 1, 2, 3, ... (later
        layers win where masks overlap). Written in the image's original orientation
        via the stored ``voxtell_props``, with a JSON sidecar legend.
        """
        image_layer = self.selected_image_layer
        if image_layer is None:
            show_warning("Select the source image in 'Image Selection' first.")
            return

        layers = [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, Labels)
            and layer.metadata.get("voxtell_source") == image_layer.name
            and layer.metadata.get("voxtell_props") is not None
        ]
        if not layers:
            show_warning(
                f"No VoxTell segmentations found for '{image_layer.name}'. "
                "Run a segmentation first."
            )
            return

        default_stem = self.output_name_input.text().strip() or f"{image_layer.name}_voxtell"
        safe_stem = "".join(c if c.isalnum() or c in " -_" else "_" for c in default_stem)
        safe_stem = safe_stem.strip().replace(" ", "_") or "segmentation"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save segmentations as multi-label NIfTI",
            f"{safe_stem}.nii.gz",
            "NIfTI (*.nii.gz)",
        )
        if not path:
            return
        if not path.endswith((".nii", ".nii.gz")):
            path += ".nii.gz"

        # Merge every layer's labels into one map with fresh sequential values.
        combined = np.zeros(tuple(np.asarray(image_layer.data).shape), dtype=np.uint16)
        legend = {}
        value = 0
        for layer in layers:
            data = np.asarray(layer.data)
            layer_legend = layer.metadata.get("voxtell_legend", {1: layer.name})
            for src_value, name in layer_legend.items():
                value += 1
                combined[data == src_value] = value
                legend[value] = name

        props = layers[0].metadata["voxtell_props"]
        try:
            NibabelIOWithReorient().write_seg(combined, path, props)
        except Exception as e:  # noqa: BLE001 - surface any write failure to the user
            show_error(f"Failed to save: {e}")
            return

        # Sidecar legend mapping label value -> prompt name.
        stem = path[:-7] if path.endswith(".nii.gz") else os.path.splitext(path)[0]
        with open(f"{stem}.json", "w") as f:
            json.dump({str(k): v for k, v in legend.items()}, f, indent=2)

        show_info(
            f"Saved {len(legend)} labels for '{image_layer.name}' to "
            f"{os.path.basename(path)}: " + ", ".join(f"{k}={v}" for k, v in legend.items())
        )
