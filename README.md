# napari-voxtell

A napari plugin for text-promptable segmentation.

## Description

napari-voxtell is a simplified napari plugin that provides text-based prompting for image segmentation. 
Instead of complex interaction tools like lasso, point, or bounding box selections, users can simply 
type a text description of what they want to segment.

## Features

- Simple text-based prompting interface
- Image layer selection
- Random mask generation (placeholder for AI integration)
- Easy to extend with your own AI backend

## Installation

You can install `napari-voxtell` via source:

```bash
git clone https://git.dkfz.de/mic/personal/group1/personal-projects/napari-voxtell
cd napari-voxtell
pip install -e .
```

## Usage

1. Open napari
2. Load an image
3. Open the Voxtell widget from the Plugins menu
4. Select your image layer
5. Enter a text prompt describing what you want to segment
6. Click Submit or press Enter

## Extending

The current implementation generates a random mask as a placeholder. To integrate your own AI model:

1. Modify the `on_submit` method in `widget_main.py`
2. Replace the random mask generation with your AI model inference
3. Pass the text prompt to your model and generate the segmentation

Example:

```python
def on_submit(self):
    text = self.text_input.text()
    image_layer = self.selected_image_layer
    
    # Your AI model here
    # segmentation = your_model.predict(image_layer.data, text)
    
    # For now, just a placeholder
    segmentation = np.random.randint(0, 2, size=image_layer.data.shape, dtype=np.uint8)
```

## License

Distributed under the terms of the Apache Software License 2.0 license.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
