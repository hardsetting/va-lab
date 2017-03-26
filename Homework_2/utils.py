import numpy as np


def otsu_threshold(img):
    # Compute normalized histogram with smoothing
    hist = np.bincount(img.flatten(), minlength=256) + 1
    norm_hist = hist.astype(float) / np.sum(hist)

    best_t = 0
    max_sb = 0
    for t in range(1, 255):
        q1 = np.sum(norm_hist[:t])
        q2 = np.sum(norm_hist[t:])

        mu1 = np.sum(np.arange(0, t) * norm_hist[:t]) / q1
        mu2 = np.sum(np.arange(t, 256) * norm_hist[t:]) / q2

        sb = q1*q2*(mu1 - mu2)**2
        if sb > max_sb:
            max_sb = sb
            best_t = t

    return best_t


def conv(img, kernel):

    sy, sx = img.shape
    by, bx = np.array(kernel.shape).astype(int) / 2

    res = np.zeros_like(img)
    for x in range(bx, sx-bx):
        for y in range(by, sy-by):
            res[y, x] = np.sum(img[y-by:y+by+1, x-bx:x+bx+1]*kernel)

    return res


def gaussian(k):
    s = 1 if k <= 7 else 2 if k <= 13 else 3
    rng_c = np.abs(np.arange(k).reshape(-1, 1) - k // 2)
    rng_e = np.exp(-rng_c**2 / (2*s**2))
    gauss = np.matmul(rng_e, rng_e.T) / (np.sqrt(2 * np.pi) * s)
    return gauss / np.sum(gauss)
