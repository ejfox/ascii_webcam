import cv2
import numpy as np
from time import time
from ultralytics import YOLO

# ASCII characters for face, from lightest to darkest
# ASCII_CHARS = '       .:-=+*#%@█▓▒░▄▀■□▪▫▬▭▮▯▰▱▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿'
ASCII_CHARS = '       ......::::-=+*#KLXM%@#0X'

def apply_contrast(image, factor):
    # Increase the effect of the contrast adjustment
    return np.clip((image - 128) * factor + 128, 0, 255).astype(np.uint8)

def apply_dithering(image):
    # Use traditional 80s style dithering
    h, w = image.shape
    dither_matrix = np.array([[0, 128], [192, 64]])
    dither_tile = np.tile(dither_matrix, (h//2 + 1, w//2 + 1))[:h, :w]
    return np.clip((image // 128) * 255 + dither_tile, 0, 255).astype(np.uint8)

def gray_to_ascii(gray):
    # Adjust thresholds for more extreme contrast
    thresholds = [0, 25, 50, 100, 150, 200, 225, 255]
    ascii_levels = [ASCII_CHARS[i] for i in range(0, len(ASCII_CHARS), len(ASCII_CHARS) // len(thresholds))]
    
    result = np.zeros_like(gray, dtype=object)
    for i in range(len(thresholds)):
        if i == 0:
            mask = gray <= thresholds[i]
        elif i == len(thresholds) - 1:
            mask = gray > thresholds[i-1]
        else:
            mask = (gray > thresholds[i-1]) & (gray <= thresholds[i])
        result[mask] = ascii_levels[i]
    return result

def gray_to_ascii(gray):
    # Adjust thresholds for more extreme contrast
    thresholds = [0, 25, 50, 100, 150, 200, 225, 255]
    ascii_levels = [ASCII_CHARS[i] for i in range(0, len(ASCII_CHARS), len(ASCII_CHARS) // len(thresholds))]
    
    result = np.zeros_like(gray, dtype=object)
    for i in range(len(thresholds)):
        if i == 0:
            mask = gray <= thresholds[i]
        elif i == len(thresholds) - 1:
            mask = gray > thresholds[i-1]
        else:
            mask = (gray > thresholds[i-1]) & (gray <= thresholds[i])
        result[mask] = ascii_levels[i]
    return result

def m(contrast_factor=1, dither=True):
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n.pt")
    
    frame_count, fps, last_time = 0, 0, time()
    width, height = 80, 40
    
    prev_gray = None
    smooth_factor = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces
        results = model(frame, classes=[0])  # 0 is the class index for person in COCO dataset
        
        if len(results[0].boxes) > 0:
            # Get the first detected face
            box = results[0].boxes[0].xyxy.cpu().numpy()[0].astype(int)
            x1, y1, x2, y2 = box
            
            # Extract face area
            face_area = frame[y1:y2, x1:x2]
        else:
            # If no face detected, use the whole frame
            face_area = frame
        
        # Resize and convert to grayscale
        gray = cv2.cvtColor(cv2.resize(face_area, (width, height)), cv2.COLOR_BGR2GRAY)
        
        # Apply contrast
        gray = apply_contrast(gray, contrast_factor)
        
        # Apply dithering if enabled
        if dither:
            gray = apply_dithering(gray)
        
        # Apply smoothing
        if prev_gray is not None:
            gray = (smooth_factor * prev_gray + (1 - smooth_factor) * gray).astype(np.uint8)
        
        prev_gray = gray
        
        # Convert grayscale to ASCII
        ascii_frame = gray_to_ascii(gray)
        
        # Clear screen and print ASCII art
        print("\033[H\033[J", end="")
        for row in ascii_frame:
            print(''.join(row))
        
        # Print settings box
        settings_box = f"╔════════════════════════════════════╗\n"
        settings_box += f"║ Contrast: {contrast_factor:<6.2f} | Dither: {str(dither):<5} ║\n"
        settings_box += f"║ FPS: {fps:<6.2f}                       ║\n"
        settings_box += f"╚════════════════════════════════════╝"
        print(settings_box)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            current_time = time()
            fps = 10 / (current_time - last_time)
            last_time = current_time

    cap.release()

# Call the function with desired contrast and dithering settings
m(contrast_factor=2, dither=True)