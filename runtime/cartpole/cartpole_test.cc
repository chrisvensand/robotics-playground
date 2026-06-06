#include "runtime/cartpole/cartpole.h"

#include "gtest/gtest.h"

TEST(CartPoleTest, ResetReturnsZeros) {
    cartpole::CartPole env;
    auto obs = env.reset();
    ASSERT_EQ(obs.size(), 4);
    for (float v : obs) {
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

TEST(CartPoleTest, StepChangesVelocity) {
    cartpole::CartPole env;
    env.reset();
    auto step = env.step({1.0f});
    EXPECT_NE(step.observation[1], 0.0f);
}

TEST(CartPoleTest, EpisodeEnds) {
    cartpole::CartPole env;
    env.reset();
    sim::Step step;
    for (int i = 0; i < 500; i++) {
        step = env.step({1.0f});
    }
    EXPECT_TRUE(step.done);
}
