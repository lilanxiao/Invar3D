import fps_np_ext
import numpy as np
import time

def farthest_point_sample(points:np.array, num_sample:int):
    """farthest point sampling

    Arguments:
        points {np.array} -- points to be sampled (n,3)
        num_sample {int} -- numbers of samples

    Returns:
        list of int -- index of sampled points
    """
    assert points.ndim == 2
    assert points.shape[1] == 3, "only support 3 dimension points"
    p = np.ascontiguousarray(points.astype(np.float32))
    return fps_np_ext.fps(p, num_sample)

def test_fps(round:int=20):
    total = 0
    for _ in range(round):
        a = np.random.rand(20000,3).astype(np.float32)
        start = time.time()
        idx = farthest_point_sample(a, 1000)
        t = (time.time()-start)*1000.
        print("... %.3f ms"%t)
        total += t
    print("average: %.3f ms"%(total/round))

def vis():
    import matplotlib.pyplot as plt
    a = np.random.randn(1000, 3)
    a[:,2] = 0
    idx = farthest_point_sample(a, 200)
    plt.scatter(a[:,0], a[:, 1])
    plt.scatter(a[idx,0], a[idx, 1])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    test_fps()
    vis()