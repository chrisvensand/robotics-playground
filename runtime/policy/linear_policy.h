#pragma once
#include <string>
#include <vector>

namespace policy {

class LinearPolicy {
   public:
    static LinearPolicy load(const std::string& path);
    std::vector<float> act(const std::vector<float>& observation) const;
    int observation_dim() const { return obs_dim_; }
    int action_dim() const { return action_dim_; }

   private:
    int obs_dim_ = 0;
    int action_dim_ = 0;
    std::vector<float> weights_;
};

}  // namespace policy
