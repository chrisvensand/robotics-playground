#include "runtime/cartpole/cartpole.h"

#include <cmath>

namespace cartpole {

namespace {
constexpr float kGravity = 9.8f;
constexpr float kCartMass = 1.0f;
constexpr float kPoleMass = 0.1f;
constexpr float kPoleLength = 0.5f;
constexpr float kForceMag = 10.0f;
constexpr float kTau = 0.02f;
constexpr float kThetaThreshold = 12.0f * 3.14159265f / 180.0f;
constexpr float kXThreshold = 2.4f;
constexpr int kMaxSteps = 500;
}  // namespace

CartPole::CartPole() { reset(); }

std::vector<float> CartPole::reset() {
    x_ = x_dot_ = theta_ = theta_dot_ = 0.0f;
    step_count_ = 0;
    return {x_, x_dot_, theta_, theta_dot_};
}

sim::Step CartPole::step(const std::vector<float>& action) {
    float force = (action[0] > 0.0f) ? kForceMag : -kForceMag;
    float costheta = std::cos(theta_);
    float sintheta = std::sin(theta_);
    float total_mass = kCartMass + kPoleMass;
    float pole_ml = kPoleMass * kPoleLength;
    float temp = (force + pole_ml * theta_dot_ * theta_dot_ * sintheta) / total_mass;
    float thetaacc = (kGravity * sintheta - costheta * temp) /
                     (kPoleLength * (4.0f / 3.0f - kPoleMass * costheta * costheta / total_mass));
    float xacc = temp - pole_ml * thetaacc * costheta / total_mass;
    x_ += kTau * x_dot_;
    x_dot_ += kTau * xacc;
    theta_ += kTau * theta_dot_;
    theta_dot_ += kTau * thetaacc;
    step_count_++;
    bool done = std::abs(x_) > kXThreshold || std::abs(theta_) > kThetaThreshold ||
                step_count_ >= kMaxSteps;
    return {{x_, x_dot_, theta_, theta_dot_}, 1.0f, done};
}

}  // namespace cartpole
