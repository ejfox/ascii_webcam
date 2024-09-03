import cv2
import numpy as np

def webcam_to_ascii(frame, cols=80, rows=40):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame
    cell_width = frame.shape[1] // cols
    cell_height = frame.shape[0] // rows
    resized = cv2.resize(gray, (cols, rows))
    
    # Define ASCII characters
    ascii_chars = ' .:-=+*#%@'
    
    # Convert to ASCII
    ascii_frame = ''
    for row in resized:
        for pixel in row:
            # Ensure the index is within the range of ascii_chars
            index = min(int(pixel / 28), len(ascii_chars) - 1)  # 255 / 9 ≈ 28
            ascii_frame += ascii_chars[index]
        ascii_frame += '\n'
    
    return ascii_frame

def main():
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        ascii_frame = webcam_to_ascii(frame)
        print("\033[H\033[J", end="")  # Clear console
        print(ascii_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()