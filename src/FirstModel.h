#ifndef FIRSTMODEL_H
#define FIRSTMODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include "Model.h"

class FirstModel : public Model {
    public:
        FirstModel(int n, float lambda);
        virtual void train(const std::vector<std::string>& sentence);
        float hypothesis(int j, const std::vector<int>& phrase);
        float error(const std::vector<int>& phrase);
        float derivTheta(int l, int j, const std::vector<int>& phrase);
        float derivWord(int c, int k, const std::vector<int>& phrase);
        int getWordInd(const std::string& word);
        void gradCheck(const std::vector<int>& phrase);

    private:
        int _n;
        float _lambda;
        std::vector<std::vector<float>> _theta;
        std::vector<std::vector<float>> _indtovec;
        std::unordered_map<std::string, int> _wordtoind;
};

#endif

