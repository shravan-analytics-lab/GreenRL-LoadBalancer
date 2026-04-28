import numpy as np

def generate_traffic():
    time = np.arange(0, 24)
    base = 50
    pattern = 30 * np.sin((time - 6) / 24 * 2 * np.pi)
    noise = np.random.randint(-10, 10, size=24)
    traffic = base + pattern + noise
    return traffic