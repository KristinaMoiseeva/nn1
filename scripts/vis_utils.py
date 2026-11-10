import numpy as np


def visualize_grid(Xs, ubound=255.0, padding=1):
    n, h, w, channels = Xs.shape
    size = int(np.ceil(np.sqrt(n)))
    grid = np.zeros((size * h + (size - 1) * padding,
                     size * w + (size - 1) * padding, channels))
    for i, image in enumerate(Xs):
        low, high = image.min(), image.max()
        image = ubound * (image - low) / max(float(high - low), 1e-12)
        row, col = divmod(i, size)
        r, c = row * (h + padding), col * (w + padding)
        grid[r:r+h, c:c+w] = image
    return grid
