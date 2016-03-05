#include "KMeans.h"

KMeans::KMeans(SecondModel* model) :
    _model(model) {
}
#include <iostream>
std::vector<int> KMeans::group(int k, Distances::fun dist, std::vector<int> indices) {
    std::vector<int> clusters(indices.size(),0);
    std::vector<Eigen::VectorXf> middles;
    bool modif = true;
    std::normal_distribution<float> distr(0,1);
    std::mt19937 gen;
    int dim = (_model->getVector(0)).size();
    for(int i = 0; i < k; ++i) {
        Eigen::VectorXf vec(dim);
        for (int j = 0; j < dim; ++j) {
            vec(j) = distr(gen);
        }
        middles.push_back(vec);
    }
    int iter = 0;
    while(modif) {
        if (iter < _ITERMAX) {
            modif = false;
            for (int i = 0; i < indices.size(); ++i) {
                Eigen::VectorXf vec = _model->getVector(indices[i]);
                int ind = 0;
                float min = dist(vec, middles[0]);
                for (int j = 0; j < k; ++j) {
                    if (dist(vec, middles[j]) < min) {
                        ind = j;
                        min = dist(vec, middles[j]);
                    }
                }
                if (clusters[i] != ind) {
                    clusters[i] = ind;
                    modif = true;
                }
            }
            for (int j = 0; j < k; ++j) {
                Eigen::VectorXf middle = Eigen::VectorXf::Zero(dim);
                int classsize = 0;
                for (int i = 0; i <indices.size(); ++i) {
                    if (clusters[i] == j) {
                        middle += _model->getVector(indices[i]);
                        classsize += 1;
                    }
                }
                middle /= classsize;
                middles[j] = middle;
                iter++;
            }
        }
        else {
            break;
        }
    }
    return clusters;
}
