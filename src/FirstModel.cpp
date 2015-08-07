#include "FirstModel.h"
#include "Serializer.h"
#include <random>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <algorithm>

//Magic, untested values
static const float RANDOM_MEAN = 0.0;
static const float RANDOM_STDDEV = 10.0;

static float carre(float x) {
    return x*x;
}


FirstModel::FirstModel(int n, float lambda) :
    _n(n),
    _lambda(lambda),
    _theta(Eigen::MatrixXf::Zero(4*_n, _n)) {

    std::normal_distribution<float> dist(RANDOM_MEAN, RANDOM_STDDEV);
    for(int i = 0; i < 4*n; ++i) {
        for(int j = 0; j < n; ++j) {
            _theta(i, j) = dist(_gen);
        }
    }
}

void FirstModel::train(const std::vector<std::string>& sentence, float alpha) {
    std::vector<int> indices = wordsToIndices(sentence);
    Eigen::MatrixXf dtheta(4*_n, _n);
    std::vector<Eigen::VectorXf> dword(sentence.size(), Eigen::VectorXf::Zero(_n));
    for(int i = 0; i<4*_n; i++) {
        dtheta.row(i) = derivTheta(i, indices).transpose();
    }
    for(int i = 0; i < sentence.size()-4; i++) {
        dword[i] = derivWord(i, indices);
    }
    _theta -= dtheta*alpha;
    for (int i = 0; i<sentence.size(); i++) {
        _indtovec[indices[i]] -= dword[i]*alpha;
    }
}

float FirstModel::error(const std::vector<std::string>& sentence) {
    return error(wordsToIndices(sentence));
}

int FirstModel::vocabSize() {
    return _indtovec.size();
}

void FirstModel::save(const std::string& file) {
    Serializer s;
    s.initWrite(file);
    s.writeInt(_n);
    s.writeFloat(_lambda);
    s.writeMat(_theta);
    s.writeInt(_indtovec.size());
    for(const Eigen::VectorXf& v : _indtovec) {
        s.writeVec(v);
    }
    s.writeInt(_wordtoind.size());
    for(const std::pair<std::string, int>& p : _wordtoind) {
        s.writeString(p.first);
        s.writeInt(p.second);
    }
}

void FirstModel::load(const std::string& file) {
    Serializer s;
    s.initRead(file);
    _n = s.readInt();
    _lambda = s.readFloat();
    _theta = s.readMat();
    int size = s.readInt();
    for(int i = 0; i < size; ++i) {
        _indtovec.push_back(s.readVec());
    }
    size = s.readInt();
    for(int i = 0; i < size; ++i) {
        std::string word = s.readString();
        int ind = s.readInt();
        _wordtoind.emplace(word, ind);
    }
}


Eigen::VectorXf FirstModel::hypothesis(const std::vector<int>& phrase) const {
    Eigen::VectorXf words(4*_n);
    words << _indtovec[phrase[0]], _indtovec[phrase[1]], _indtovec[phrase[3]], _indtovec[phrase[4]];
    return _theta.transpose()*words;
}

float FirstModel::error(const std::vector<int>& sentence) const {
    float S = 0;
    for(int i = 0; i < sentence.size()-4; ++i) {
        std::vector<int> example(sentence.begin()+i, sentence.begin()+i+5);
        S += errorEx(example);
    }
    S /= sentence.size()-4;

    //Regularization
    for(int j = 0; j < _n; ++j) {
        for (int l = 0; l<4*_n; l++) {
            S += _lambda/2 * carre(_theta(l, j));
        }
        for (int i = 0; i<sentence.size(); i++) {
            S += _lambda/2 * carre(_indtovec[sentence[i]][j]);
        }
    }
    return S;
}

Eigen::VectorXf FirstModel::derivTheta(int l, const std::vector<int>& sentence) const {
    Eigen::VectorXf res = Eigen::VectorXf::Zero(_n);
    for(int i = 0; i < sentence.size()-4; ++i) {
        std::vector<int> example(sentence.begin()+i, sentence.begin()+i+5);
        res += derivThetaEx(l, example);
    }
    return res/(sentence.size()-4)+ _lambda*_theta.row(l).transpose();
}

Eigen::VectorXf FirstModel::derivWord(int i, const std::vector<int>& sentence) const {
    Eigen::VectorXf res = Eigen::VectorXf::Zero(_n);
    for(int c = 0; c < 5; c++) {
        if(i-c >= 0 && i-c+5 <= sentence.size()) {
            std::vector<int> example(sentence.begin()+i-c, sentence.begin()+i-c+5);
            res += derivWordEx(c, example);
        }
    }
    return res/(sentence.size()-4) + _lambda*_indtovec[sentence[i]];
}

int FirstModel::getWordInd(const std::string& word) {
    auto it = _wordtoind.find(word);
    if(it != _wordtoind.end()) {
        return it->second;
    } else {
        int id = _indtovec.size();
        _wordtoind.emplace(word, id);
        Eigen::VectorXf vec(_n);
        std::normal_distribution<float> dist(RANDOM_MEAN, RANDOM_STDDEV);
        for(int i = 0; i < _n; ++i) {
            vec(i) = dist(_gen);
        }
        _indtovec.push_back(vec);
        return id;
    }
}

std::vector<int> FirstModel::wordsToIndices(const std::vector<std::string>& sentence) {
    std::vector<int> indices(sentence.size());
    std::transform(sentence.begin(),sentence.end(),indices.begin(),[this](const std::string& s) {return this->getWordInd(s);});
    return indices;
}


void FirstModel::gradCheck(const std::vector<int>& sentence) {
    float epsilon = sqrt(FLT_EPSILON);
    printf("Theta:\n");
    for(int l = 0; l < 4*_n; ++l) {
        for(int j = 0; j < _n; ++j) {
            float theta = _theta(l, j);
            _theta(l, j) = theta + epsilon;
            float errpe = error(sentence);
            _theta(l, j) = theta - epsilon;
            float errme = error(sentence);
            _theta(l, j) = theta;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivTheta(l, sentence)(j);
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
            float calc = derivWord(c, sentence)(k);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, c, k);
        }
        printf("\n");
    }
}

float FirstModel::errorEx(const std::vector<int>& example) const {
    float S = 0;
    for (int j = 0; j<_n; j++) {
        S += carre(hypothesis(example)(j) - _indtovec[example[2]][j]);
    }
    return S/2;
}

Eigen::VectorXf FirstModel::derivThetaEx(int l, const std::vector<int>& example) const {
    int c = l/_n;
    c += (c>=2);
    int k = l%_n;
    return (hypothesis(example) -_indtovec[example[2]])*_indtovec[example[c]][k];
}

Eigen::VectorXf FirstModel::derivWordEx(int c, const std::vector<int>& example) const {
    if (c == 2) {
        return _indtovec[example[c]]-hypothesis(example);
    } else {
        return _theta.block((c-(c>2))*_n, 0, _n, _n) * (hypothesis(example) - _indtovec[example[2]]);
    }
}


void FirstModel::gradCheckEx(const std::vector<int>& example) {
    float epsilon = sqrt(FLT_EPSILON);
    printf("Theta:\n");
    for(int l = 0; l < 4*_n; ++l) {
        for(int j = 0; j < _n; ++j) {
            float theta = _theta(l, j);
            _theta(l, j) = theta + epsilon;
            float errpe = errorEx(example);
            _theta(l, j) = theta - epsilon;
            float errme = errorEx(example);
            _theta(l, j) = theta;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivThetaEx(l, example)(j);
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
            float calc = derivWordEx(c, example)(k);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, c, k);
        }
        printf("\n");
    }
}

