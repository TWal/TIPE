#ifndef FIRSTMODEL_H
#define FIRSTMODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include "Model.h"

class FirstModel : public Model {
    public:
        FirstModel(int n, float lambda);
        virtual void train(const std::vector<std::string>& sentence, float alpha);
        virtual float error(const std::vector<std::string>& sentence);
        virtual int vocabSize();

        float hypothesis(int j, const std::vector<int>& phrase);
        float error(const std::vector<int>& sentence);
        float derivTheta(int l, int j, const std::vector<int>& sentence);
        float derivWord(int i, int k, const std::vector<int>& sentence);
        int getWordInd(const std::string& word);
        std::vector<int> wordsToIndices(const std::vector<std::string>& sentence);
        void gradCheck(const std::vector<int>& phrase);

        //Functions working on a single example (5 words) without regularization
        float errorEx(const std::vector<int>& example);
        float derivThetaEx(int l, int j, const std::vector<int>& example);
        float derivWordEx(int c, int k, const std::vector<int>& example);
        void gradCheckEx(const std::vector<int>& example);

    private:
        int _n;
        float _lambda;
        std::vector<std::vector<float>> _theta;
        std::vector<std::vector<float>> _indtovec;
        std::unordered_map<std::string, int> _wordtoind;
};

#endif

