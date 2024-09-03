import cv2, numpy as np
from time import time
from scipy.ndimage import gaussian_filter

B = "⠀⠄⠆⠖⠶⠿"  # Reduced set for higher contrast

def perlin(shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    shape = np.array(shape)
    scales = shape / scale
    
    noise = np.zeros(shape)
    for octave in range(octaves):
        freq = lacunarity**octave
        amp = persistence**octave
        noise += amp * gaussian_filter(np.random.rand(*shape), sigma=1/freq) * scales

    return (noise - noise.min()) / (noise.max() - noise.min())

def g(f, c=80, r=40):
    gray = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (c, r))
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # Increase contrast
    return np.vectorize(lambda x: B[min(int(x/(256/len(B))), len(B)-1)])(gray)

def m():
    c = cv2.VideoCapture(0)
    p = perlin((40, 80), scale=1, seed=42)
    t, fps, frame_time = 0, 0, time()
    glitch_lines = np.zeros(40, dtype=int)
    
    while True:
        _, f = c.read()
        
        # Update Perlin noise
        if t % 10 == 0:
            p = np.roll(p, 1, axis=1)  # Slow shift
        
        a = g(f)
        
        # Apply Perlin-based glitch
        mask = p > np.percentile(p, 90)
        a[mask] = B[-1]
        
        # Datamoshing-like effect
        if t % 30 == 0:
            glitch_lines = np.random.randint(0, 40, 5)  # Select 5 random lines
        for line in glitch_lines:
            shift = np.random.randint(-10, 11)
            a[line] = np.roll(a[line], shift)
        
        print("\033[H\033[J" + '\n'.join(''.join(row) for row in a))
        print(f"FPS: {fps:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        t += 1
        if t % 10 == 0:
            new_time = time()
            fps = 10 / (new_time - frame_time)
            frame_time = new_time

    c.release()

m()