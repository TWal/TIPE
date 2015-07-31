#ifndef FIRSTMODEL_H
#define FIRSTMODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include "Model.h"

class FirstModel : public Model {
    public:
        FirstModel(int n, float lambda);
        virtual void train(const std::vector<std::string>& sentence, float alpha);
        virtual float error(const std::vector<std::string>& sentence);
        virtual int vocabSize();

        Eigen::VectorXf hypothesis(const std::vector<int>& phrase);
        float error(const std::vector<int>& sentence);
        Eigen::VectorXf derivTheta(int l, const std::vector<int>& sentence);
        Eigen::VectorXf derivWord(int i, const std::vector<int>& sentence);
        int getWordInd(const std::string& word);
        std::vector<int> wordsToIndices(const std::vector<std::string>& sentence);
        void gradCheck(const std::vector<int>& phrase);

        //Functions working on a single example (5 words) without regularization
        float errorEx(const std::vector<int>& example);
        Eigen::VectorXf derivThetaEx(int l, const std::vector<int>& example);
        Eigen::VectorXf derivWordEx(int c, const std::vector<int>& example);
        void gradCheckEx(const std::vector<int>& example);

    private:
        int _n;
        float _lambda;
        Eigen::MatrixXf _theta;
        std::vector<Eigen::VectorXf> _indtovec;
        std::unordered_map<std::string, int> _wordtoind;
};

#endif

