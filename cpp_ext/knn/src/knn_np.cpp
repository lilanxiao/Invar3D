#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "knn.h"
#include <pybind11/stl.h>

namespace py = pybind11;

std::pair<py::array_t<float>, py::array_t<long>> batch_knn_wrapper(py::array_t<float> query, py::array_t<float> source, int k, bool parallel){
    py::buffer_info query_info = query.request();
    float *q_data = static_cast<float*>(query_info.ptr);
    std::vector<ssize_t> q_shape = query_info.shape;

    py::buffer_info source_info = source.request();
    float *s_data = static_cast<float*>(source_info.ptr);
    std::vector<ssize_t> s_shape = source_info.shape;

    std::vector<ssize_t> o_shape = {q_shape[0], q_shape[1], k};

    /* No pointer is passed, so NumPy will allocate the buffer */
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
    auto inds = py::array_t<long>(o_shape, 
                            {o_shape[1]*o_shape[2]*sizeof(long), o_shape[2]*sizeof(long), sizeof(long)});       // index
    py::buffer_info inds_info = inds.request();
    long *buf = (long *) inds_info.ptr;

    if (parallel)
        cpp_knn_batch_omp(s_data, s_shape[0], s_shape[1], s_shape[2], q_data, q_shape[1], k, buf);
    else
        cpp_knn_batch(s_data, s_shape[0], s_shape[1], s_shape[2], q_data, q_shape[1], k, buf);

    std::vector<ssize_t> nn_shape = {q_shape[0], q_shape[1], k, q_shape[2]};
    auto nn = py::array_t<float>(nn_shape,
                            {nn_shape[1]*nn_shape[2]*nn_shape[3]*sizeof(float), 
                            nn_shape[2]*nn_shape[3]*sizeof(float),
                            nn_shape[3]*sizeof(float),
                            sizeof(float)});        // features
    py::buffer_info nn_info = nn.request();
    float *nn_buf = (float*) nn_info.ptr;

    for (int batch=0; batch < q_shape[0]; ++batch){
        for (int i=0; i<q_shape[1]; ++i){
            for (int j=0; j<k; ++j){
                int offset = buf[batch*o_shape[1]*o_shape[2] + i*o_shape[2] + j];
                memcpy(nn_buf + batch*nn_shape[1]*nn_shape[2]*nn_shape[3] + i*nn_shape[2]*nn_shape[3] + j*nn_shape[3], 
                s_data + batch * s_shape[1]*s_shape[2] + offset * s_shape[2],
                s_shape[2] * sizeof(float));
            }
        }
    }
    return {nn, inds};
}

std::pair<py::array_t<float>, py::array_t<long>> knn_wrapper(py::array_t<float> query, py::array_t<float> source, int k, bool parallel){
    py::buffer_info query_info = query.request();
    float *q_data = static_cast<float*>(query_info.ptr);
    std::vector<ssize_t> q_shape = query_info.shape;

    py::buffer_info source_info = source.request();
    float *s_data = static_cast<float*>(source_info.ptr);
    std::vector<ssize_t> s_shape = source_info.shape;

    std::vector<ssize_t> o_shape = {q_shape[0], k};

    /* No pointer is passed, so NumPy will allocate the buffer */
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
    auto inds = py::array_t<long>(o_shape, 
                            {o_shape[1]*sizeof(long), sizeof(long)});       // index
    py::buffer_info inds_info = inds.request();
    long *buf = (long *) inds_info.ptr;

    if (parallel)
        cpp_knn_omp(s_data, s_shape[0], s_shape[1], q_data, q_shape[0], k, buf);
    else
        cpp_knn(s_data, s_shape[0], s_shape[1], q_data, q_shape[0], k, buf);

    std::vector<ssize_t> nn_shape = {q_shape[0], k, q_shape[1]};
    auto nn = py::array_t<float>(nn_shape,
                            {nn_shape[1]*nn_shape[2]*sizeof(float), 
                            nn_shape[2]*sizeof(float),
                            sizeof(float)});        // features
    py::buffer_info nn_info = nn.request();
    float *nn_buf = (float*) nn_info.ptr;

    for (int i=0; i<q_shape[0]; ++i){
        for (int j=0; j<k; ++j){
            int offset = buf[i*o_shape[1] + j];
            memcpy(nn_buf + i*nn_shape[1]*nn_shape[2] + j*nn_shape[2], 
            s_data + offset * s_shape[1],
            s_shape[1] * sizeof(float));
        }
    }
    return {nn, inds};
}


std::pair<py::array_t<float>, py::array_t<long>> radius_knn_wrapper(py::array_t<float> query, py::array_t<float> source, float radius, int k){
    py::buffer_info query_info = query.request();
    float *q_data = static_cast<float*>(query_info.ptr);
    std::vector<ssize_t> q_shape = query_info.shape;

    py::buffer_info source_info = source.request();
    float *s_data = static_cast<float*>(source_info.ptr);
    std::vector<ssize_t> s_shape = source_info.shape;

    std::vector<ssize_t> o_shape = {q_shape[0], k};

    auto inds = py::array_t<long>(o_shape, 
                            {o_shape[1]*sizeof(long), sizeof(long)});       // index
    py::buffer_info inds_info = inds.request();
    long *buf = (long *) inds_info.ptr;

    cpp_knn_radius(s_data, s_shape[0], s_shape[1], q_data, q_shape[0], k, radius, buf);

    std::vector<ssize_t> nn_shape = {q_shape[0], k, q_shape[1]};
    auto nn = py::array_t<float>(nn_shape,
                            {nn_shape[1]*nn_shape[2]*sizeof(float), 
                            nn_shape[2]*sizeof(float),
                            sizeof(float)});        // features
    py::buffer_info nn_info = nn.request();
    float *nn_buf = (float*) nn_info.ptr;

    for (int i=0; i<q_shape[0]; ++i){
        for (int j=0; j<k; ++j){
            int offset = buf[i*o_shape[1] + j];
            memcpy(nn_buf + i*nn_shape[1]*nn_shape[2] + j*nn_shape[2], 
            s_data + offset * s_shape[1],
            s_shape[1] * sizeof(float));
        }
    }
    return {nn, inds};
}

std::pair<py::array_t<float>, py::array_t<long>> radius_knn2_wrapper(py::array_t<float> query, py::array_t<float> source, float radius, int k){
    py::buffer_info query_info = query.request();
    float *q_data = static_cast<float*>(query_info.ptr);
    std::vector<ssize_t> q_shape = query_info.shape;

    py::buffer_info source_info = source.request();
    float *s_data = static_cast<float*>(source_info.ptr);
    std::vector<ssize_t> s_shape = source_info.shape;

    std::vector<ssize_t> o_shape = {q_shape[0], k};

    auto inds = py::array_t<long>(o_shape, 
                            {o_shape[1]*sizeof(long), sizeof(long)});       // index
    py::buffer_info inds_info = inds.request();
    long *buf = (long *) inds_info.ptr;

    cpp_knn_radius2(s_data, s_shape[0], s_shape[1], q_data, q_shape[0], k, radius, buf);

    std::vector<ssize_t> nn_shape = {q_shape[0], k, q_shape[1]};
    auto nn = py::array_t<float>(nn_shape,
                            {nn_shape[1]*nn_shape[2]*sizeof(float), 
                            nn_shape[2]*sizeof(float),
                            sizeof(float)});        // features
    py::buffer_info nn_info = nn.request();
    float *nn_buf = (float*) nn_info.ptr;

    for (int i=0; i<q_shape[0]; ++i){
        for (int j=0; j<k; ++j){
            int offset = buf[i*o_shape[1] + j];
            memcpy(nn_buf + i*nn_shape[1]*nn_shape[2] + j*nn_shape[2], 
            s_data + offset * s_shape[1],
            s_shape[1] * sizeof(float));
        }
    }
    return {nn, inds};
}


PYBIND11_MODULE(knn_np_ext, m) {
    m.def("batch_knn", &batch_knn_wrapper, py::return_value_policy::automatic,
        py::arg("query"), py::arg("source"), py::arg("k"), py::arg("parallel"));
    m.def("knn", &knn_wrapper, py::return_value_policy::automatic,
        py::arg("query"), py::arg("source"), py::arg("k"), py::arg("parallel"));
    m.def("knn_radius", &radius_knn_wrapper, py::return_value_policy::automatic,
        py::arg("query"), py::arg("source"), py::arg("radius"), py::arg("k"));
    m.def("knn_radius2", &radius_knn2_wrapper, py::return_value_policy::automatic,
        py::arg("query"), py::arg("source"), py::arg("radius"), py::arg("k"));
}
