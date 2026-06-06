#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "runtime/policy/linear_policy.h"

namespace py = pybind11;

PYBIND11_MODULE(linear_policy_py, m) {
    py::class_<policy::LinearPolicy>(m, "LinearPolicy")
        .def_static("load", &policy::LinearPolicy::load)
        .def("act", &policy::LinearPolicy::act)
        .def("observation_dim", &policy::LinearPolicy::observation_dim)
        .def("action_dim", &policy::LinearPolicy::action_dim);
}
