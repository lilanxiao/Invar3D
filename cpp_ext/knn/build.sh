c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` -Iinclude src/knn_np.cpp src/knn.cpp -o knn_np_ext`python3-config --extension-suffix`