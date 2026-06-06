#pragma once
#include <vector>

namespace sim {

struct Step {
    std::vector<float> observation;
    float reward;
    bool done;
};

class Env {
   public:
    virtual ~Env() = default;
    virtual std::vector<float> reset() = 0;
    virtual Step step(const std::vector<float>& action) = 0;
    virtual int observation_dim() const = 0;
    virtual int action_dim() const = 0;
};

}  // namespace sim
