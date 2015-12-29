#include "SecondModel.h"

#include "Serializer.h"
#include <cmath>
#include <cfloat>
#include <fstream>

static float sigmoid(float x) {
    return 1/(1+exp(-x));
    // 1/2 + 1/2tanh(x/2)
}

SecondModel::SecondModel(int n, VocabManager* vocabmgr, int nsTableSize, float nsTablePower) :
    _n(n), _vocabmgr(vocabmgr), _nsTablePower(nsTablePower) {

    std::uniform_real_distribution<float> dist(-0.5/n, 0.5/n);
    std::mt19937 gen;

    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        Eigen::VectorXf vec(_n);
        _w1.push_back(Eigen::VectorXf::Zero(_n));
        _w2.push_back(Eigen::VectorXf::Zero(_n));
        for(int j = 0; j < _n; ++j) {
            _w1[i](j) = dist(gen);
            _w2[i](j) = 0;
        }
    }

    _buildNsTable(nsTableSize);
}

void SecondModel::train(const std::vector<std::string>& sentence, float alpha) {
    std::array<int, SecondModel::CTX_SIZE> ctx;
    int answer;
    sentenceToContext(sentence, ctx, answer);
    std::vector<int> wrong = negSample (answer);
    Eigen::VectorXf dW1 = derivW1Neg(ctx, answer, wrong);
    Eigen::VectorXf dW2 = derivW2(ctx, answer, true);
    std::vector<Eigen::VectorXf> dwrongW2;
    for(int i = 0; i < wrong.size(); ++i) {
        dwrongW2.push_back(derivW2(ctx, wrong[i], false));
    }

    if(false) {
        checkDerivative(&_w2[answer], dW2, ctx, answer, wrong);
        for(int i = 0; i < wrong.size(); ++i) {
            checkDerivative(&_w2[wrong[i]], dwrongW2[i], ctx, answer, wrong);
        }
        for(int i = 0; i < SecondModel::CTX_SIZE; ++i) {
            checkDerivative(&_w1[ctx[i]], dW1, ctx, answer, wrong);
        }
    }

    _w2[answer] -= alpha*dW2;
    for(int i = 0; i < wrong.size(); ++i) {
        _w2[wrong[i]] -= alpha*dwrongW2[i];
    }
    for(int i = 0; i < SecondModel::CTX_SIZE; ++i) {
        _w1[ctx[i]] -= alpha*dW1;
    }
}

int SecondModel::sentenceSize() {
    return SecondModel::CTX_SIZE+1;
}

void SecondModel::save(const std::string& file) {
    Serializer s;
    s.initWrite(file);
    s.writeInt(_n);
    _vocabmgr->save(s);
    for(const Eigen::VectorXf& v : _w1) {
        s.writeVec(v);
    }
    for(const Eigen::VectorXf& v : _w2) {
        s.writeVec(v);
    }
    s.writeInt(_nsTable.size());
}

void SecondModel::load(const std::string& file) {
    Serializer s;
    s.initRead(file);
    _n = s.readInt();
    _vocabmgr->load(s);
    _w1.clear();
    _w2.clear();
    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        _w1.push_back(s.readVec());
    }
    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        _w2.push_back(s.readVec());
    }
    _buildNsTable(s.readInt());
}

void SecondModel::displayState(const std::vector<std::string>& sentence) {
    std::array<int, SecondModel::CTX_SIZE> ctx;
    int answer;
    sentenceToContext(sentence, ctx, answer);
    printf("error negsample: %f\terror softmax:%f\twords: %d\n", errorNegSample(ctx, answer, negSample(answer)), errorSoftmax(ctx, answer), _vocabmgr->getVocabSize());
}

Eigen::VectorXf SecondModel::hidden(const std::array<int, SecondModel::CTX_SIZE>& ctx) {
    Eigen::VectorXf h = Eigen::VectorXf::Zero(_n);
    for(int i = 0; i < SecondModel::CTX_SIZE; ++i) {
        h += _w1[ctx[i]];
    }
    return h / SecondModel::CTX_SIZE;
}

float SecondModel::hypothesis(const Eigen::VectorXf& h, int i) {
    return h.dot(_w2[i]);
}

float SecondModel::error(const std::array<int, SecondModel::CTX_SIZE>& ctx, int i, bool ok) {
    Eigen::VectorXf h = hidden(ctx);
    if(ok) {
        return -log(sigmoid(hypothesis(h, i)));
    } else {
        return -log(sigmoid(-hypothesis(h, i)));
    }
}


float SecondModel::errorNegSample(const std::array<int, SecondModel::CTX_SIZE>& ctx, int answer, const std::vector<int>& wrong) {
    float result = error(ctx, answer, true);
    for(int i : wrong) {
        result += error(ctx, i, false);
    }
    return result;
}

float SecondModel::errorSoftmax(const std::array<int, SecondModel::CTX_SIZE>& ctx, int answer) {
    Eigen::VectorXf h = hidden(ctx);
    std::vector<double> res;
    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        res.push_back(hypothesis(h, i));
    }
    double max = *std::max_element(res.begin(), res.end());
    double sum = 0;
    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        res[i] = exp(res[i] - max);
        sum += res[i];
    }
    return -log(res[answer]/sum);
}


Eigen::VectorXf SecondModel::derivW1(const std::array<int, SecondModel::CTX_SIZE>& ctx, int i, bool ok) {
    Eigen::VectorXf h = hidden(ctx);
    Eigen::VectorXf eh = (sigmoid(hypothesis(h, i))-ok)*_w2[i];
    return eh/CTX_SIZE;
}

Eigen::VectorXf SecondModel::derivW1Neg(const std::array<int, SecondModel::CTX_SIZE>& ctx, int answer, const std::vector<int>& wrong) {
    Eigen::VectorXf dW1 = derivW1(ctx, answer, true);
    for(int i = 0; i < wrong.size(); i++) {
        dW1 += derivW1(ctx, wrong[i], false);
    }
    return dW1;
}

Eigen::VectorXf SecondModel::derivW2(const std::array<int, SecondModel::CTX_SIZE>& ctx, int i, bool ok) {
    Eigen::VectorXf h = hidden(ctx);
    return (sigmoid(hypothesis(h, i))-ok)*h;
}

void SecondModel::checkDerivative(Eigen::VectorXf* vector, const Eigen::VectorXf& deriv, const std::array<int, SecondModel::CTX_SIZE>& ctx, int answer, const std::vector<int> wrong) {
    float epsilon = 1e-2;
    Eigen::VectorXf& vec = *vector;
    Eigen::VectorXf approx = Eigen::VectorXf::Zero(vec.size());
    for(int i = 0; i < vec.size(); ++i) {
        float tmp = vec[i];
        vec[i] = tmp + epsilon;
        float errpe = errorNegSample(ctx, answer, wrong);
        vec[i] = tmp - epsilon;
        float errme = errorNegSample(ctx, answer, wrong);
        vec[i] = tmp;
        float approximate = (errpe-errme)/(2*epsilon);
        approx[i] = approximate;
    }
    float dot = deriv.dot(approx)/sqrt(deriv.dot(deriv)*approx.dot(approx));
    if(fabs(1-dot) >= 1e-2 || dot != dot) {
        printf("checkDerivative: dot = %f\n", dot);
    }
}


void SecondModel::sentenceToContext(const std::vector<std::string>& sentence, std::array<int, SecondModel::CTX_SIZE>& ctx, int& answer) {
    int j = 0;
    for(int i = 0; i < sentence.size(); ++i) {
        if(i == sentence.size()/2) {
            answer = _vocabmgr->getIndex(sentence[i]);
        } else {
            ctx[j] = _vocabmgr->getIndex(sentence[i]);
            ++j;
        }
    }
}

std::vector<int> SecondModel::negSample(int word) {
    std::uniform_int_distribution<int> dist = std::uniform_int_distribution<int>(0, _nsTable.size()-1);
    std::vector<int> result;
    for(int i = 0; i < 10; ++i) {
        int nb = _nsTable[dist(_gen)];
        if(nb != word) {
            result.push_back(nb);
        } else {
            --i;
        }
    }
    return result;
}

std::string SecondModel::closestWord(Eigen::VectorXf vect, std::function<float(const Eigen::VectorXf&, const Eigen::VectorXf&)> distance, bool useW1) {
    std::vector<Eigen::VectorXf>* vectors = useW1 ? &_w1 : &_w2;
    float min = distance(vect, (*vectors)[0]);
    int ind = 0;
    for(int i = 1; i < _vocabmgr->getVocabSize(); ++i) {
        float dist = distance(vect, (*vectors)[i]);
        if (dist < min) {
            min = dist;
            ind = i;
        }
    }
    return _vocabmgr->getWord(ind);
}

Eigen::VectorXf SecondModel::getVector(int ind, bool useW1) {
    if(useW1) {
        return _w1[ind];
    } else {
        return _w2[ind];
    }
}

void SecondModel::checkAccuracy(const std::string& filename, std::function<float(const Eigen::VectorXf&, const Eigen::VectorXf&)> distance, bool useW1) {
    std::ifstream file;
    file.open(filename.c_str());
    int totaltotal = 0;
    int totalaccurate = 0;
    while(!file.eof()) {
        int accurate = 0;
        int total = 0;
        std::string categoryName;
        std::getline(file, categoryName);
        while(file.peek() != ':' && !file.eof()) {
            std::array<std::string, 4> words;
            for(int i = 0; i < 4; ++i) {
                std::getline(file, words[i], i < 3 ? ' ' : '\n');
                std::transform(words[i].begin(), words[i].end(), words[i].begin(), ::tolower);
            }
            std::array<Eigen::VectorXf, 4> vecs;
            bool stop = false;
            for(int i = 0; i < 4; ++i) {
                int ind = _vocabmgr->getIndex(words[i]);
                if(ind < 0) {
                    stop = true;
                    break;
                }
                vecs[i] = getVector(_vocabmgr->getIndex(words[i]), useW1);
            }
            if(stop) {
                continue;
            }
            if(closestWord(-vecs[0] + vecs[1] + vecs[2], distance, useW1) == words[3]) {
                accurate += 1;
            }
            total += 1;
        }
        printf("%s : %d/%d, %f%%\n", categoryName.c_str(), accurate, total, float(accurate)/total*100);
        totaltotal += total;
        totalaccurate += accurate;
    }
    printf("Global results:  %d/%d, %f%%\n", totalaccurate, totaltotal, float(totalaccurate)/totaltotal*100);
}

void SecondModel::_buildNsTable(int nsTableSize) {
    _nsTable.clear();
    float sum = 0;
    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        sum += pow(_vocabmgr->getCount(i), _nsTablePower);
    }

    float proportionSum = 0;
    int tableSum = 0;
    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        proportionSum += pow(_vocabmgr->getCount(i), _nsTablePower)/sum;
        while(tableSum < proportionSum*nsTableSize && tableSum < nsTableSize) {
            _nsTable.push_back(i);
            tableSum += 1;
        }
    }
}

