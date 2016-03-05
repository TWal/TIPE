#include "Utils.h"

namespace Utils {

Eigen::MatrixXf findTransitionMatrix(const std::vector<Eigen::VectorXf>& x, const std::vector<Eigen::VectorXf>& y) {
    assert(x.size() == y.size());
    int m = x.size();
    int n = x[0].size();
    Eigen::MatrixXf mat(n, n);
    Eigen::MatrixXf X(m, n);
    Eigen::MatrixXf Y(m, n);
    for(int i = 0; i < m; ++i) {
        assert(x[i].size() == n && n == y[i].size());
        X.row(i) = x[i].transpose();
        Y.row(i) = y[i].transpose();
    }
    Eigen::MatrixXf Xpinv = (X.transpose() * X).inverse() * X.transpose();

    for(int i = 0; i < n; ++i) {
        mat.row(i) = Xpinv * Y.col(i);
    }

    return mat;
}

std::vector<int> kmeans(int k, Distances::fun dist, const std::vector<Eigen::VectorXf>& vecs, int itermax) {
    std::vector<int> clusters(vecs.size(), 0);
    std::vector<Eigen::VectorXf> middles;
    bool modif = true;
    std::normal_distribution<float> distr(0,1);
    std::mt19937 gen;
    int dim = vecs[0].size();
    for(int i = 0; i < k; ++i) {
        Eigen::VectorXf vec(dim);
        for (int j = 0; j < dim; ++j) {
            vec(j) = distr(gen);
        }
        middles.push_back(vec);
    }
    int iter = 0;
    while(modif && (itermax == -1 || iter < itermax)) {
        modif = false;
        iter++;
        for (int i = 0; i < vecs.size(); ++i) {
            Eigen::VectorXf vec = vecs[i];
            int ind = 0;
            float min = dist(vec, middles[0]);
            for (int j = 0; j < k; ++j) {
                float d = dist(vec, middles[j]);
                if (d < min) {
                    ind = j;
                    min = d;
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
            for (int i = 0; i < vecs.size(); ++i) {
                if (clusters[i] == j) {
                    middle += vecs[i];
                    classsize += 1;
                }
            }
            middle /= classsize;
            middles[j] = middle;
        }
    }
    return clusters;
}

}

