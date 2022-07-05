#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array_t<int> farthest_point_sampling(py::array_t<float> points, int nsamples){
    py::buffer_info points_info = points.request();
    float* p_data = static_cast<float*>(points_info.ptr);
    std::vector<ssize_t> p_shape = points_info.shape;
    int npoints = p_shape[0];

    std::vector<ssize_t> i_shape = {nsamples};
    auto inds = py::array_t<int>(i_shape, {sizeof(int)});

    py::buffer_info i_info = inds.request();
    int *i_data = (int *) i_info.ptr;

    for (int i=0; i<nsamples; ++i){
        i_data[i] = 0;
    }
    
    std::vector<float> temp(npoints, 1e10);

    int old = 0;
    for (int j=1; j<nsamples; ++j){
        int besti = 0;
        float best = -1;
        float x1 = p_data[old*3 + 0];
        float y1 = p_data[old*3 + 1];
        float z1 = p_data[old*3 + 2];
        for (int k=0; k<npoints; ++k){
            float x2, y2, z2;
            x2 = p_data[k*3 + 0];
            y2 = p_data[k*3 + 1];
            z2 = p_data[k*3 + 2];
            float d = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
            float d2 = std::min(d, temp[k]);    // update temp with current point
            temp[k] = d2;                       // min dist from all sampled points to *k*-th point in a point set 
            besti = d2 > best ? k : besti;      // find the largest among all distance
            best = d2 > best ? d2 : best;
        }
        old = besti;
        i_data[j] = old;
    }

    return inds;
}

PYBIND11_MODULE(fps_np_ext, m){
    m.def("fps", &farthest_point_sampling, py::arg("points").noconvert(), 
    py::arg("n_samples"), py::return_value_policy::automatic);
}