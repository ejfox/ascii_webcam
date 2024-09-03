import cv2
import numpy as np
from time import time

# B = "⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿"
B = "⠀⠈⠐⠠⡀⡈⡐⡠⢀⢈⢐⢠⣀⣈⣐⣠⣼⣾⣿"

def g(f, c=80, r=40):
    gray = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (c, r))
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # Increase contrast
    return np.vectorize(lambda x: B[min(int(x/(256/len(B))), len(B)-1)])(gray)

def blend_frames(prev, curr, blend_factor):
    prev_indices = np.vectorize(B.index)(prev)
    curr_indices = np.vectorize(B.index)(curr)
    blended_indices = (blend_factor * prev_indices + (1 - blend_factor) * curr_indices).astype(int)
    return np.vectorize(lambda x: B[min(x, len(B)-1)])(blended_indices)

def m():
    c = cv2.VideoCapture(0)
    t, fps, frame_time = 0, 0, time()
    prev_frame = None
    blend_factor = 0.8  # Adjust this to control the blending intensity
    
    while True:
        ret, f = c.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_frame = g(f)
        
        if prev_frame is None:
            prev_frame = current_frame
        
        # Datamoshing-like effect: blend current frame with previous frame
        blended_frame = blend_frames(prev_frame, current_frame, blend_factor)
        
        # Introduce some random glitches
        if t % 30 == 0:
            glitch_mask = np.random.random(blended_frame.shape) > 0.95
            blended_frame[glitch_mask] = np.random.choice(list(B))
        
        print("\033[H\033[J" + '\n'.join(''.join(row) for row in blended_frame))
        print(f"FPS: {fps:.2f}")
        
        prev_frame = blended_frame  # Update previous frame for next iteration
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        t += 1
        if t % 10 == 0:
            new_time = time()
            fps = 10 / (new_time - frame_time)
            frame_time = new_time

    c.release()

m()