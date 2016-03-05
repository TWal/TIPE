#ifndef KMEANS_H
#define KMEANS_H

#include "SecondModel.h"
#include "Distances.h"

class KMeans {
    public:
        KMeans(SecondModel* model);
        std::vector<int> group(int k, Distances::fun dist, std::vector<int> indices);
    private:
        SecondModel* _model;
        int _ITERMAX = 1000000;
};

#endif
