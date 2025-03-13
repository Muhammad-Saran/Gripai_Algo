# GR1P AI algo (Next Gen Scanner)

## Overview
Gripai_Algo is a Python-based tool designed to measure hand metrics, specifically the length and width of a hand, using advanced computer vision techniques. The project leverages the MediaPipe library for hand landmark detection and OpenCV for image processing.

A coin with a manually drawn red circle serves as a reference object with a known diameter (assumed to be a US quarter, 24.26 mm) to scale the hand measurements from pixels to real-world units (millimeters and inches). The green ellipse is overlaid on the red circle to visualize the reference scale, enabling accurate calculation of hand dimensions. The final goal is to determine the hand's length and width, categorized using a weighted average system.

The tool processes an input image of a hand with a coin, detects hand landmarks, uses the red circle to establish a scale, and outputs the processed image with annotations, along with a debug image showing the red circle mask.

## Features
- Detects hand landmarks using MediaPipe.
- Uses a red circle drawn around a coin as a reference scale to calculate hand metrics.
- Measures hand length and width in inches, scaled by the coin's known diameter.
- Outputs processed images with green ellipse overlays and hand measurement annotations.
- Generates a debug image to visualize the red circle detection process.

## Installation and Setup

### Prerequisites
- Python 3.10
- Git (for cloning the repository)
- Basic knowledge of command-line operations

### Step-by-Step Guide

#### Clone the Repository
Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/your-username/gripai-scanner-algo.git
cd gripai-scanner-algo
```

#### Create a Virtual Environment with Python 3.10
Create and activate a virtual environment to isolate dependencies:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```
Ensure you have Python 3.10 installed. If not, download it from [python.org](https://www.python.org) or use a version manager like pyenv.

#### Install Dependencies
The `requirements.txt` file is already included in the repository. Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

#### Prepare Test Image
Place a test image (e.g., `test_image.jpg`) in the repository root. The image should contain a hand with a coin and a red circle drawn around the coin to serve as the reference scale.

#### Run the Test Script
Execute the `test.py` script to process the image and generate outputs:
```bash
python test.py
```
The script will prompt you to enter the path to your test image (e.g., `test_image.jpg`). Upon successful execution, it will:
- Generate a processed image (`processed_images/processed_hand_image_with_coin_and_landmarks.jpg`) with hand landmarks, a green ellipse overlaying the red circle, and hand length/width annotations.
- Create a debug image (`debug_red_mask.jpg`) showing the isolated red circle mask.

## Usage

### Input Requirements
- The input image must contain a hand with a coin.
- A red circle must be manually drawn around the coin to serve as the reference scale.
- The coin's real-world diameter is assumed to be **24.26 mm** (a US quarter). Adjust `get_coin_diameter()` if using a different coin.

### Output
- **Processed Image**: Saved in the `processed_images` directory, showing:
  - Hand landmarks (drawn by MediaPipe).
  - A green ellipse overlaying the red circle (reference scale).
  - Text annotations for hand length and width in inches.
- **Debug Image**: Saved as `debug_red_mask.jpg`, displaying the binary mask used to detect the red circle.

### Example Workflow
1. Place `test_image.jpg` in the repository.
2. Run `python test.py` and input `test_image.jpg` when prompted.
3. Check the `processed_images` and root directories for the output files.

## Code Structure
- `detect_hand_region()`: Identifies the hand region using MediaPipe landmarks.
- `detect_hand()`: Draws hand landmarks on the image.
- `detect_coin()`: Detects the red circle, fits an ellipse, and establishes the scale.
- `get_hand_metrics()`: Calculates hand length and width in pixels.
- `process_hand_scan_image()`: Orchestrates the image processing pipeline and computes scaled hand measurements.
- `main()`: Handles user input and initiates processing.

## Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the tool. Suggestions for enhancing red circle detection, supporting multiple coin sizes, or refining hand measurement accuracy are welcome.

## License
This project is licensed under the **MIT License** - see the `LICENSE` file for details.

## Acknowledgments
- **MediaPipe** for hand landmark detection.
- **OpenCV** for image processing capabilities.
- **Pillow** for image handling.

## Additional Instructions

### Creating `test.py`
Create a `test.py` file in the repository root with the following content to simplify testing. Replace `hand_measurement` with the name of your main script file (e.g., `hand_measurement.py` if you save your code there):
```python
from hand_measurement import process_hand_scan_image

if __name__ == "__main__":
    process_hand_scan_image("test_image.jpg")  # Replace with your test image path
```

### Test Image
Prepare a test image (e.g., `test_image.jpg`) with a hand, a coin, and a red circle around the coin, and place it in the repository.

### Running the Code
Follow the step-by-step guide to set up the environment and run `test.py`. The processed image will show hand length and width, and the debug image will help verify the red circle detection.

