#ifndef SECONDMODEL_H
#define SECONDMODEL_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unordered_map>
#include <random>
#include <array>
#include "Model.h"

class SecondModel : public Model {
    public:
        SecondModel(int n, int v);
        virtual void train(const std::vector<std::string>& sentence, float alpha);
        virtual int sentenceSize();
        virtual void save(const std::string& file);
        virtual void load(const std::string& file);
        virtual void displayState(const std::vector<std::string>& sentence);

        static const int CTX_SIZE = 4;
        Eigen::VectorXf hidden(const std::array<int, SecondModel::CTX_SIZE>& ctx);
        float hypothesis(const Eigen::VectorXf& h, int i);
        float error(const std::array<int, SecondModel::CTX_SIZE>& ctx, int i, bool ok);
        float errorNegSample(const std::array<int, SecondModel::CTX_SIZE>& ctx, int answer);
        float errorSoftmax(const std::array<int, SecondModel::CTX_SIZE>& ctx, int answer);
        Eigen::VectorXf derivW1(const std::array<int, SecondModel::CTX_SIZE>& ctx, int i, bool ok);
        Eigen::VectorXf derivW2(const std::array<int, SecondModel::CTX_SIZE>& ctx, int i, bool ok);
        void gradCheck(const std::array<int, SecondModel::CTX_SIZE>& ctx, int j, int k);
        void sentenceToContext(const std::vector<std::string>& sentence, std::array<int, SecondModel::CTX_SIZE>& ctx, int& answer);
        std::vector<int> negSample(int word);
        int getWordInd(const std::string& word);

    private:
        int _n;
        int _v;
        std::vector<Eigen::VectorXf> _w1;
        std::vector<Eigen::VectorXf> _w2;
        std::unordered_map<std::string, int> _wordtoind;
        std::mt19937 _gen;
};

#endif
