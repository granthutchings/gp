/*
 * Copyright (c) 2017, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

#include "test_functions.hpp"
#include <iostream>

namespace gp {
namespace test {

  // A simple function (quadratic with lots of bumps) on the interval [0, 1].
  double BumpyParabola(double x, double freq, double amp) {
    return (x - 0.5) * (x - 0.5) + amp * sin(2.0 * M_PI * freq * x);
  }

  double Surf(double x, double y) {
    return (1+2*x+pow(x,2.0))*(1+2*y+pow(y,2.0));
  }

  // Some high dimensional functions
  double P01(std::vector<double> v) {

		double c = 1/v.size();
		double w = .3;
		double f = 2*M_PI*w;
		for (int i=0;i<v.size();i++) {
			f = f + c * v[i];
		}
		f = cos(f);

		return f;
  }

	double P02(std::vector<double> v) {
		
		double c = 1/v.size();
		double f = 1;
		for (int i=0;i<v.size();i++) {
			f = f + c * v[i];
		}
		f = 1/pow(v.size()+1,f);

		return f;
	}
	double P03(std::vector<double> v) {
		
		double c = 1/v.size();
		double w = .5;
		double f = 0;
		for (int i=0;i<v.size();i++) {
			f = f + pow(2.0,c) * pow(2.0,v[i]-w);
		}
		f = exp(-f);

		return f;
	}	
	double P04(std::vector<double> v) {
		
		double c = 2.0;
		double w = .5;
		double f = 0;
		for (int i=0;i<v.size();i++) {
			f = f + c * std::abs( v[i] - w );
		}
		f = exp(-f);

		return f;
	}
  double highDFunction(std::vector<double> v) {
    
		double temp = 0;
    double z = 0;
    for (int i = 0; i < v.size(); i++) {
		temp = pow(-1.0,static_cast<double>(i))*v[i]*v[i]; // (-1)^n * v[i]^2
		z += temp;
  	}

    return z;
  }

} //\namespace test
} //\namespace gp
