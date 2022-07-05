import numpy as np
from knn_np_ext import batch_knn, knn, knn_radius, knn_radius2

def batch_knn_flann(query, source, k:int, parallel:bool=True):
    """KNN for batched data

    Args:
        query (numpy.array): float32, (B, M, dims)
        source (numpy.array): float32, (B, N, dims)
        k (int): number of neighbors
        parallel (bool, optional): use openmp to accelerate. Defaults to True.

    Returns:
        numpy.array: float32, coordinates of KNN
        numpy.array: long, index of KNN
    """
    assert query.ndim == 3
    assert source.ndim == 3
    return batch_knn(np.ascontiguousarray(query.astype(np.float32)), 
                    np.ascontiguousarray(source.astype(np.float32)), 
                    int(k), parallel)

def knn_flann(query, source, k:int, parallel:bool=True):
    """KNN without batch

    Args:
        query (numpy.array): float32, (M, dims)
        source (numpy.array): float32, (N, dims)
        k (int): number of neighbors
        parallel (bool, optional): use openmp to accelerate. Defaults to True.

    Returns:
        numpy.array: float32, coordinates of KNN
        numpy.array: long, index of KNN
    """
    assert query.ndim == 2
    assert source.ndim == 2
    return knn(np.ascontiguousarray(query.astype(np.float32)), 
                np.ascontiguousarray(source.astype(np.float32)), 
                int(k), parallel)


def ball_query(query, source, radius:float, k:int):
    """find points within a radius. note the points are not necessarily the nearest ones.

    Args:
        query (numpy.array): float32, (M, dims)
        source (numpy.array): float32, (N, dims)
        radius (float): max radius of search
        k (int): number of neighbors

    Returns:
        numpy.array: float32, coordinates of KNN
        numpy.array: long, index of KNN
    """
    assert query.ndim == 2
    assert source.ndim == 2
    return knn_radius(np.ascontiguousarray(query.astype(np.float32)), 
                np.ascontiguousarray(source.astype(np.float32)), 
                float(radius),
                int(k))


def ball_query2(query, source, radius:float, k:int):
    """find all points with in radius and evenly sample k.

    Args:
        query (numpy.array): float32, (M, dims)
        source (numpy.array): float32, (N, dims)
        radius (float): max radius of search
        k (int): number of neighbors

    Returns:
        numpy.array: float32, coordinates of KNN
        numpy.array: long, index of KNN
    """
    assert query.ndim == 2
    assert source.ndim == 2
    return knn_radius2(np.ascontiguousarray(query.astype(np.float32)), 
                np.ascontiguousarray(source.astype(np.float32)), 
                float(radius),
                int(k))


if __name__ == "__main__":
    import time
    s = np.random.rand(16, 200000, 3)
    q = np.random.rand(16, 50000, 3)
    start = time.time()    
    nbs, ind = batch_knn_flann(q, s, 5, parallel=True)
    end = time.time() - start
    print(nbs.shape, nbs.dtype)
    print(ind.shape, ind.dtype)
    print(end)
    
    
