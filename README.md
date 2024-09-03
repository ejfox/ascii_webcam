# ASCII Webcam Art

This repository contains a collection of Python scripts that transform your webcam feed into various forms of ASCII art. Each script offers a unique take on rendering your camera input as text-based visual art.

## Scripts Overview

### 1. ascii_webcam_eyes.py

This script focuses on detecting and displaying eyes as emojis in an ASCII canvas.

Key features:
- Uses OpenCV for face and eye detection
- Displays detected eyes as eye emojis (👁️)
- Maintains a consistent 15 FPS for smooth performance
- Mirrors the webcam input for a more intuitive experience

### 2. ascii_yolo_webcam.py

This script provides a more detailed ASCII representation of the entire face, with special emphasis on eyes.

Key features:
- Uses YOLO (You Only Look Once) for face detection
- Employs OpenCV for eye detection within the face region
- Renders the face in ASCII characters with contrast and dithering options
- Highlights detected eyes in red
- Displays performance metrics and detection information

### 3. glitch_braille_cam.py

This script creates an artistic, glitch-like effect using Braille characters.

Key features:
- Renders the webcam feed using Braille characters
- Implements a datamoshing-like effect by blending frames
- Introduces random glitches for an artistic touch
- Provides a unique, abstract representation of the video feed

## Key Differences

1. **Detection Methods**:
    - `ascii_webcam_eyes.py` uses simple OpenCV detection
    - `ascii_yolo_webcam.py` combines YOLO and OpenCV for more robust detection
    - `glitch_braille_cam.py` doesn't use specific feature detection

2. **Rendering Style**:
    - `ascii_webcam_eyes.py` focuses solely on rendering eyes
    - `ascii_yolo_webcam.py` renders the entire face in ASCII with eye highlights
    - `glitch_braille_cam.py` uses Braille characters for a more abstract representation

3. **Performance**:
    - `ascii_webcam_eyes.py` is optimized for performance, maintaining 15 FPS
    - `ascii_yolo_webcam.py` is more computationally intensive due to YOLO and detailed rendering
    - `glitch_braille_cam.py` balances between artistic effect and performance

4. **Artistic Approach**:
    - `ascii_webcam_eyes.py` is minimalistic, focusing on eye representation
    - `ascii_yolo_webcam.py` aims for a more complete and detailed ASCII portrait
    - `glitch_braille_cam.py` takes an abstract, glitch-art approach

## Usage

To run any of the scripts, ensure you have the required dependencies installed, then execute the script using Python:

```
python script_name.py
```

Press 'q' to exit the program.

Enjoy exploring these different approaches to ASCII webcam art!