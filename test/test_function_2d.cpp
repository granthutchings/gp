#include <kernels/rbf_kernel.hpp>
#include <process/gaussian_process.hpp>
#include <optimization/cost_functors.hpp>
#include <utils/types.hpp>
#include <utils/plot_1d.hpp>

#include "test_functions.hpp"
#include "test_plotting.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <random>
#include <vector>
#include <math.h>


#ifdef SYSTEM_OSX
#include <GLUT/glut.h>
#endif

#ifdef SYSTEM_LINUX
//#include <GL/glew.h>
#include <GL/glut.h>
#endif

DECLARE_bool(visualize);
DECLARE_bool(verbose);

namespace gp {
namespace test {

// Check that we can approximate a simple 2D function.
TEST(GaussianProcess, TestFunctionApprox2D) {
  const size_t kNumTrainingPoints = 100;
  const size_t kNumTestPoints = 100;
  const double kMaxRmsError = 0.1;
  const double kNoiseVariance = 1e-3;
  const double kLength = 1.0;

  const size_t kBatchSize = 16;
  const size_t kGradUpdates = 10000;
  //const size_t kRelearnInterval = 2000;
  const double kStepSize = 1.0;

  // Random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // Get training points/targets.
  PointSet points(new std::vector<VectorXd>);
  VectorXd targets(kNumTrainingPoints);
	
	for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    points->push_back(VectorXd::Constant(2, unif(rng)));
    targets(ii) = unif(rng);
  }	

	// Train a GP.
  const Kernel::Ptr kernel = RbfKernel::Create(VectorXd::Constant(2, kLength));
  GaussianProcess gp(kernel, kNoiseVariance, points, targets,
                     kNumTrainingPoints);

	// Run gradient descent.
  std::vector<VectorXd> batch_points;
  std::vector<double> batch_targets;
  double mse = std::numeric_limits<double>::infinity();
  for (size_t ii = 0; ii < kGradUpdates; ii++) {
    // Maybe relearn hyperparameters.
    //if (ii % kRelearnInterval == kRelearnInterval - 1)
    //  EXPECT_TRUE(gp.LearnHyperparams());

    // Get a random batch.
    batch_points.clear();
    batch_targets.clear();

    for (size_t jj = 0; jj < kBatchSize; jj++) {
      const double x = unif(rng);
			const double y = unif(rng);
			VectorXd batch_point(2);
			point.push_back(x);
			point.push_back(y);
      batch_points.push_back(batch_point);
      batch_targets.push_back(Surf(x,y));
    }

		// Update parameters.
    mse = gp.UpdateTargets(batch_points, batch_targets,
                           kStepSize, ii == kGradUpdates - 1);

    if (FLAGS_verbose && ii % 100 == 1)
      std::printf("MSE at step %zu was %5.3f.\n", ii, mse);
	}

	// Test that we have approximated the function well.
  double squared_error = 0.0;
  double mean, variance;
  for (size_t ii = 0; ii < kNumTestPoints; ii++) {
    const double x = unif(rng);
		const double y = unif(rng);
    VectorXd test_point(2);
		test_point.push_back(x);
		test_point.push_back(y);
    gp.Evaluate(test_point, mean, variance);
    squared_error += (mean - Surf(x,y)) * (mean - Surf(x,y));
  }

  EXPECT_LE(std::sqrt(squared_error / static_cast<double>(kNumTestPoints)),
            kMaxRmsError);

}

} //\namespace test
} //\namespace gp
