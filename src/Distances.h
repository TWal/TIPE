#ifndef DISTANCES_H
#define DISTANCES_H

#include <Eigen/Dense>

namespace Distances {

typedef std::function<float(const Eigen::VectorXf&, const Eigen::VectorXf&)> fun;
float euclidian(const Eigen::VectorXf& vect_one, const Eigen::VectorXf& vect_two);
float cosinus(const Eigen::VectorXf& vect_one, const Eigen::VectorXf& vect_two);

}

#endif

