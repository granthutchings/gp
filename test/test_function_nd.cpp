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
#include <iostream>


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

// How well can we approximate data in higher-dimentions
TEST(GaussianProcess, TestFunctionApproxND) {
	const int dim = 8;
  const size_t kNumTrainingPoints = 100;
  const size_t kNumTestPoints = 100;
  const double kMaxRmsError = 0.1;
  const double kNoiseVariance = 1e-3;
  const double kLength = 1.0;

  const size_t kBatchSize = 64;
  const size_t kGradUpdates = 10000;
  //const size_t kRelearnInterval = 2000;
  const double kStepSize = 1.0;

	double x;

  // Random number generator.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // Get training points/targets.
  PointSet points(new std::vector<VectorXd>);
  VectorXd targets(kNumTrainingPoints);
	std::vector<double> coordVector;
	VectorXd coords(dim);
	for (size_t ii = 0; ii < kNumTrainingPoints; ii++) {
    for (int i = 0; i<dim; i++) {
			x = unif(rng);
			coords(i) = x;
			coordVector.push_back(x);
		}
    points->push_back(coords);
    targets(ii) = P02(coordVector);
  }	

	// Train a GP.
  const Kernel::Ptr kernel = RbfKernel::Create(VectorXd::Constant(dim, kLength));
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

		VectorXd batch_point(dim);
		std::vector<double> batch_coords;
    for (size_t jj = 0; jj < kBatchSize; jj++) {
			batch_coords.clear();
			for (int i = 0; i<dim; i++) {
				x = unif(rng);
				batch_coords.push_back(x);
				batch_point(i) = x;
			}
      batch_points.push_back(batch_point); // push back a vectorxd representing the coords of a point
      batch_targets.push_back(P02(batch_coords)); // push back a double - function evaluated at point vector
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
	VectorXd test_point(dim);
	coordVector.clear();
  for (size_t ii = 0; ii < kNumTestPoints; ii++) {
    //std::vector<double> test_coords; // n dimentional coordinate vector
			for (size_t i = 0; i < dim; i++) {
        x = unif(rng);
				coordVector.push_back(x); // random point in R^n st each coord is on [0,1]
				test_point[i] = x;
			}
    gp.Evaluate(test_point, mean, variance);
    squared_error += (mean - P02(coordVector)) * (mean - P02(coordVector));
  }

  std::printf("		RMSE %dd: %lf\n", dim , std::sqrt(squared_error / static_cast<double>(kNumTestPoints)));
  EXPECT_LE(std::sqrt(squared_error / static_cast<double>(kNumTestPoints)),
            kMaxRmsError);

}

} //\namespace test
} //\namespace gp
