import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Set up the ASCII canvas
width, height = 80, 40

# Emoji for eyes
EYE_CHAR = '👁️'

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    eyes = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes_in_face = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes_in_face:
            eyes.append((x + ex + ew//2, y + ey + eh//2))
    
    return eyes[:2]  # Return at most 2 eyes

def m():
    frame_time = 1.0 / 15  # Target time per frame for 15 FPS
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        eyes = detect_eyes(frame)
        
        canvas = np.full((height, width), ' ', dtype=object)
        
        for eye in eyes:
            ex_scaled = int(eye[0] * width / frame.shape[1])
            ey_scaled = int(eye[1] * height / frame.shape[0])
            canvas[ey_scaled, ex_scaled] = EYE_CHAR

        print("\033[H\033[J", end="")
        for row in canvas:
            print(''.join(str(char) for char in row))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Control frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)

    cap.release()

m()