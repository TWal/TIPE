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

std::vector<int> FirstModel::wordsToIndices(const std::vector<std::string>& sentence) {
    std::vector<int> indices(sentence.size());
    std::transform(sentence.begin(),sentence.end(),indices.begin(),[this](const std::string& s) {return this->getWordInd(s);});
    return indices;
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
    std::vector<int> indices = wordsToIndices(sentence);
    std::vector<std::vector<float>> dtheta(4*_n,std::vector<float>(_n, 0.0));
    std::vector<std::vector<float>> dword(sentence.size(), std::vector<float>(_n, 0.0));
    for(int i = 0; i<4*_n; i++) {
        for(int j = 0; j<_n; j++) {
            dtheta[i][j] = derivTheta(i, j, indices);
        }
    }
    for(int i = 0; i < sentence.size()-4; i++) {
        for(int k = 0; k<_n; k++) {
            dword[i][k] = derivWord(i, k, indices);
        }
    }

    for (int i = 0; i<4*_n; i++) {
        for (int j = 0; j<_n; j++) {
            _theta[i][j] -= dtheta[i][j]*alpha;
        }
    }
    for (int i = 0; i<sentence.size(); i++) {
        for (int j = 0; j<_n; j++) {
            _indtovec[indices[i]][j] -= dword[i][j]*alpha;
        }
    }
}

float FirstModel::error(const std::vector<std::string>& sentence) {
    return error(wordsToIndices(sentence));
}

int FirstModel::vocabSize() {
    return _indtovec.size();
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

float FirstModel::error(const std::vector<int>& sentence) {
    float S = 0;
    for(int i = 0; i < sentence.size()-4; ++i) {
        std::vector<int> example(sentence.begin()+i, sentence.begin()+i+5);
        S += errorEx(example);
    }
    S /= sentence.size()-4;

    //Regularization
    for(int j = 0; j < _n; ++j) {
        for (int l = 0; l<4*_n; l++) {
            S += _lambda/2 * carre(_theta[l][j]);
        }
        for (int i = 0; i<sentence.size(); i++) {
            S += _lambda/2 * carre(_indtovec[sentence[i]][j]);
        }
    }
    return S;
}

float FirstModel::derivTheta(int l, int j, const std::vector<int>& sentence) {
    float res = 0;
    for(int i = 0; i < sentence.size()-4; ++i) {
        std::vector<int> example(sentence.begin()+i, sentence.begin()+i+5);
        res += derivThetaEx(l, j, example);
    }
    return res/(sentence.size()-4)+ _lambda*_theta[l][j];
}

float FirstModel::derivWord(int i, int k, const std::vector<int>& sentence) {
    float res = 0;
    for(int c = 0; c < 5; c++) {
        if(i-c >= 0 && i-c+5 <= sentence.size()) {
            std::vector<int> example(sentence.begin()+i-c, sentence.begin()+i-c+5);
            res += derivWordEx(c, k, example);
        }
    }
    return res/(sentence.size()-4) + _lambda*_indtovec[sentence[i]][k];
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

void FirstModel::gradCheck(const std::vector<int>& sentence) {
    float epsilon = sqrt(FLT_EPSILON);
    printf("Theta:\n");
    for(int l = 0; l < 4*_n; ++l) {
        for(int j = 0; j < _n; ++j) {
            float theta = _theta[l][j];
            _theta[l][j] = theta + epsilon;
            float errpe = error(sentence);
            _theta[l][j] = theta - epsilon;
            float errme = error(sentence);
            _theta[l][j] = theta;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivTheta(l, j, sentence);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, l, j);
        }
    }

    printf("Words:\n");
    for(int c = 0; c < sentence.size(); ++c) {
        for(int k = 0; k < _n; ++k) {
            float word = _indtovec[sentence[c]][k];
            _indtovec[sentence[c]][k] = word + epsilon;
            float errpe = error(sentence);
            _indtovec[sentence[c]][k] = word - epsilon;
            float errme = error(sentence);
            _indtovec[sentence[c]][k] = word;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivWord(c, k, sentence);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, c, k);
        }
        printf("\n");
    }
}

float FirstModel::errorEx(const std::vector<int>& example) {
    float S = 0;
    for (int j = 0; j<_n; j++) {
        S += carre(hypothesis(j,example) - _indtovec[example[2]][j]);
    }
    return S/2;
}

float FirstModel::derivThetaEx(int l, int j, const std::vector<int>& example) {
    int c = l/_n;
    c += (c>=2);
    int k = l%_n;
    return (hypothesis(j, example)-_indtovec[example[2]][j])*_indtovec[example[c]][k];
}

float FirstModel::derivWordEx(int c, int k, const std::vector<int>& example) {
    if (c == 2) {
        return _indtovec[example[c]][k]-hypothesis(k, example);
    } else {
        float S = 0;
        for (int j=0; j < _n; j++) {
            S += _theta[(c-(c>2))*_n+k][j]*(hypothesis(j,example)-_indtovec[example[2]][j]);
        }
        return S;
    }
}


void FirstModel::gradCheckEx(const std::vector<int>& example) {
    float epsilon = sqrt(FLT_EPSILON);
    printf("Theta:\n");
    for(int l = 0; l < 4*_n; ++l) {
        for(int j = 0; j < _n; ++j) {
            float theta = _theta[l][j];
            _theta[l][j] = theta + epsilon;
            float errpe = errorEx(example);
            _theta[l][j] = theta - epsilon;
            float errme = errorEx(example);
            _theta[l][j] = theta;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivThetaEx(l, j, example);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, l, j);
        }
    }

    printf("Words:\n");
    for(int c = 0; c < 5; ++c) {
        for(int k = 0; k < _n; ++k) {
            float word = _indtovec[example[c]][k];
            _indtovec[example[c]][k] = word + epsilon;
            float errpe = errorEx(example);
            _indtovec[example[c]][k] = word - epsilon;
            float errme = errorEx(example);
            _indtovec[example[c]][k] = word;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivWordEx(c, k, example);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, c, k);
        }
        printf("\n");
    }
}

