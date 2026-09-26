import os, time
import numpy as np, numba as nb

@nb.njit(parallel=True, fastmath=True)
def kern(h, w, it):
    out = np.empty((h, w), np.float32)
    for y in nb.prange(h):
        py = y / h
        for x in range(w):
            px = x / w
            a = 0.0
            for i in range(it):
                a += np.sin(px * 3.1 + i * 0.37) * np.cos(py * 2.3 - i * 0.21) * np.exp(-0.001 * i)
            out[y, x] = a
    return out

kern(8, 8, 4)
n = nb.config.NUMBA_NUM_THREADS
for th in sorted({1, n}):
    nb.set_num_threads(th)
    best = 1e9
    for _ in range(2):
        t = time.perf_counter(); kern(804, 1920, 48); best = min(best, time.perf_counter() - t)
    print(f'threads={th:3d}  best={best:.3f}s')
print('numba_threads', n, 'cpu_count', os.cpu_count())
