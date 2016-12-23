#ifndef AJC_HGUARD_RANDOM_SOURCE
#define AJC_HGUARD_RANDOM_SOURCE

#include <random>
#include <iostream>

#include "xoroshiro.hpp"

class RandomSource {
public:
  virtual double sample() = 0;
};

class DefaultRandomSource : public RandomSource {
  xoroshiro128plus_engine engine;
  std::uniform_real_distribution<double> distribution;

public:
  static DefaultRandomSource shared_source;

  DefaultRandomSource();
  double sample() override { return distribution(engine); }
};

class ConstantSource : public RandomSource {
  const double value;

public:
  ConstantSource(double x) : value(x) {}
  double sample() override { return value; }
};

#endif

