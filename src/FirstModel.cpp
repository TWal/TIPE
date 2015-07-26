#include "FirstModel.h"
#include <random>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <algorithm>

//Magic, untested values
static const float RANDOM_MEAN = 0.0;
static const float RANDOM_STDDEV = 3.0;

static float carre(float x) {
    return x*x;
}

FirstModel::FirstModel(int n, float lambda) :
    _n(n),
    _lambda(lambda),
    _theta(4*n, std::vector<float>(n, 0.0)) {

    std::mt19937 gen;
    std::normal_distribution<float> dist(RANDOM_MEAN, RANDOM_STDDEV);
    for(int i = 0; i < 4*n; ++i) {
        for(int j = 0; j < n; ++j) {
            _theta[i][j] = dist(gen);
        }
    }
}

void FirstModel::train(const std::vector<std::string>& sentence, float alpha) {
    std::vector<int> indices(sentence.size());
    std::transform(sentence.begin(),sentence.end(),indices.begin(),[this](const std::string& s) {return this->getWordInd(s);});
    std::vector<std::vector<float>> dtheta(4*_n,std::vector<float>(_n, 0.0));
    for(int i = 0; i<4*_n; i++) {
        for(int j = 0; j<_n; j++) {
            for(int k = 0; k < sentence.size()-4; k++) {
                std::vector<int> example(indices.begin()+k,indices.begin()+k+5);
                dtheta[i][j] += derivTheta(i, j, example);
            }
            dtheta[i][j] /= sentence.size()-4;
        }
    }
    std::vector<std::vector<float>> dword(sentence.size(), std::vector<float>(_n, 0.0));
    for(int i = 0; i < sentence.size()-4; i++) {
        std::vector<int> example(indices.begin()+i, indices.begin()+i+5);
        for(int c = 0; c < 5; c++) {
            for(int k = 0; k<_n; k++) {
                dword[i+c][k] += derivWord(c, k, example) / (sentence.size()-4);
            }
        }
    }
    for (int i = 0; i<4*_n; i++) {
        for (int j = 0; j<_n; j++) {
            _theta[i][j] += dtheta[i][j]*alpha;
        }
    }
    for (int i = 0; i<sentence.size(); i++) {
        for (int j = 0; j<_n; j++) {
            _indtovec[indices[i]][j] += dword[i][j]*alpha;
        }
    }
}

float FirstModel::hypothesis(int j, const std::vector<int>& phrase) {
    float S = 0;
    for (int l = 0; l < 4*_n; l++) {
        float valthet = _theta[l][j];
        int c = l/_n;
        if (c>=2) {
            c++;
        }
        int k = l%_n;
        float valvec = _indtovec[phrase[c]][k];
        S += valthet*valvec;
    }
    return S;
}

float FirstModel::error(const std::vector<int>& phrase) {
    float S = 0;
    for (int j = 0; j<_n; j++) {
        S += carre(hypothesis(j,phrase) - _indtovec[phrase[2]][j]);
        for (int l = 0; l<4*_n; l++) {
            S += _lambda * carre(_theta[l][j]);
        }
        for (int i = 0; i<5; i++) {
            S += _lambda * carre(_indtovec[phrase[i]][j]);
        }
    }
    return S/2;
}

float FirstModel::derivTheta(int l, int j, const std::vector<int>& phrase) {
    int c = l/_n;
    c += (c>=2);
    int k = l%_n;
    return (hypothesis(j, phrase)-_indtovec[phrase[2]][j])*_indtovec[phrase[c]][k] + _lambda*_theta[l][j];
}

float FirstModel::derivWord(int c, int k, const std::vector<int>& phrase) {
    if (c == 2) {
        return _indtovec[phrase[c]][k]-hypothesis(k, phrase) + _lambda*_indtovec[phrase[c]][k];
    } else {
        float S = 0;
        for (int j=0; j < _n; j++) {
            S += _theta[(c-(c>2))*_n+k][j]*(hypothesis(j,phrase)-_indtovec[phrase[2]][j]);
        }
        return S + _lambda*_indtovec[phrase[c]][k];
    }
}

int FirstModel::getWordInd(const std::string& word) {
    auto it = _wordtoind.find(word);
    if(it != _wordtoind.end()) {
        return it->second;
    } else {
        int id = _indtovec.size();
        _wordtoind.emplace(word, id);
        std::vector<float> vec;
        std::mt19937 gen;
        std::normal_distribution<float> dist(RANDOM_MEAN, RANDOM_STDDEV);
        for(int i = 0; i < _n; ++i) {
            vec.push_back(dist(gen));
        }
        _indtovec.push_back(vec);
        return id;
    }
}

void FirstModel::gradCheck(const std::vector<int>& phrase) {
    float epsilon = sqrt(FLT_EPSILON);
    printf("Theta:\n");
    for(int l = 0; l < 4*_n; ++l) {
        for(int j = 0; j < _n; ++j) {
            float theta = _theta[l][j];
            _theta[l][j] = theta + epsilon;
            float errpe = error(phrase);
            _theta[l][j] = theta - epsilon;
            float errme = error(phrase);
            _theta[l][j] = theta;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivTheta(l, j, phrase);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, l, j);
        }
    }

    printf("Words:\n");
    for(int c = 0; c < 5; ++c) {
        for(int k = 0; k < _n; ++k) {
            float word = _indtovec[phrase[c]][k];
            _indtovec[phrase[c]][k] = word + epsilon;
            float errpe = error(phrase);
            _indtovec[phrase[c]][k] = word - epsilon;
            float errme = error(phrase);
            _indtovec[phrase[c]][k] = word;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivWord(c, k, phrase);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, c, k);
        }
        printf("\n");
    }
}

