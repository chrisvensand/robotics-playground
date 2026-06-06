#pragma once
#include "runtime/sim/env.h"

namespace cartpole {

class CartPole : public sim::Env {
   public:
    CartPole();
    std::vector<float> reset() override;
    sim::Step step(const std::vector<float>& action) override;
    int observation_dim() const override { return 4; }
    int action_dim() const override { return 1; }

   private:
    float x_, x_dot_, theta_, theta_dot_;
    int step_count_;
};

}  // namespace cartpole
