#include "runtime/policy/linear_policy.h"

#include <fstream>
#include <stdexcept>

namespace policy {

LinearPolicy LinearPolicy::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    LinearPolicy p;
    f.read(reinterpret_cast<char*>(&p.obs_dim_), sizeof(int));
    f.read(reinterpret_cast<char*>(&p.action_dim_), sizeof(int));
    p.weights_.resize(p.obs_dim_ * p.action_dim_);
    f.read(reinterpret_cast<char*>(p.weights_.data()), p.weights_.size() * sizeof(float));
    return p;
}

std::vector<float> LinearPolicy::act(const std::vector<float>& obs) const {
    std::vector<float> out(action_dim_, 0.0f);
    for (int a = 0; a < action_dim_; ++a)
        for (int o = 0; o < obs_dim_; ++o) out[a] += weights_[a * obs_dim_ + o] * obs[o];
    return out;
}

}  // namespace policy
