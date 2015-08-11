#include "Distances.h"
#include <math.h>

namespace Distances {

float euclidian(const Eigen::VectorXf& vect_one, const Eigen::VectorXf& vect_two) {
    Eigen::VectorXf dist = vect_one - vect_two;
    return sqrt(dist.dot(dist));
}

float cosinus(const Eigen::VectorXf& vect_one, const Eigen::VectorXf& vect_two) {
    float norme_one = sqrt(vect_one.dot(vect_one));
    float norme_two = sqrt(vect_two.dot(vect_two));
    return 1 - vect_one.dot(vect_two) / (norme_one*norme_two);
}

}

