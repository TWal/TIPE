#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include "Distances.h"

namespace Utils {

Eigen::MatrixXf findTransitionMatrix(const std::vector<Eigen::VectorXf>& x, const std::vector<Eigen::VectorXf>& y);
std::vector<int> kmeans(int k, Distances::fun dist, const std::vector<Eigen::VectorXf>& vecs, int itermax = -1);

}

#endif

