#include <cstdio>

#include "runtime/cartpole/cartpole.h"
#include "runtime/policy/linear_policy.h"

int main() {
    cartpole::CartPole env;
    auto pol = policy::LinearPolicy::load("model/cartpole/cartpole_policy.bin");
    auto obs = env.reset();
    int steps = 0;
    float total = 0.0f;
    while (true) {
        auto action = pol.act(obs);
        auto step = env.step(action);
        obs = step.observation;
        total += step.reward;
        steps++;
        if (step.done) break;
    }
    std::printf("cartpole_runner: %d steps, reward %.1f\n", steps, total);
    return 0;
}
