#include "SecondModel.h"

#include "Serializer.h"
#include <cmath>
#include <cfloat>

static const float RANDOM_MEAN = 0.0;
static const float RANDOM_STDDEV = 1.0;

static float sigmoid(float x) {
    return 1/(1+exp(-x));
    // 1/2 + 1/2tanh(x/2)
}

SecondModel::SecondModel(int n, VocabManager* vocabmgr, int nsTableSize, float nsTablePower) :
    _n(n), _vocabmgr(vocabmgr), _nsTablePower(nsTablePower) {

    std::normal_distribution<float> dist(RANDOM_MEAN, RANDOM_STDDEV);
    std::mt19937 gen;

    for(int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        Eigen::VectorXf vec(_n);
        _w1.push_back(Eigen::VectorXf::Zero(_n));
        _w2.push_back(Eigen::VectorXf::Zero(_n));
        for(int j = 0; j < _n; ++j) {
            _w1[i](j) = dist(gen);
            _w2[i](j) = dist(gen);
        }
    }

    _buildNsTable(nsTableSize);
}

void SecondModel::train(const std::vector<std::string>& sentence, float alpha) {
    std::array<int, SecondModel::CTX_SIZE> ctx;
    int answer;
    sentenceToContext(sentence, ctx, answer);
    Eigen::VectorXf dW2 = derivW2 (ctx, answer, true);
    std::vector<int> wrong = negSample (answer);
    std::vector<Eigen::VectorXf> dwrongW2;
    Eigen::VectorXf dW1 = derivW1 (ctx, answer, true);
    for(int i = 0; i < wrong.size(); i++) {
        dwrongW2.push_back(derivW2(ctx, wrong[i], false));
        dW1 += derivW1 (ctx, wrong[i], false);
    }
    _w2[answer] -= alpha*dW2;
    for(int i = 0; i < wrong.size(); i++) {
        _w2[wrong[i]] -= alpha*dwrongW2[i];
    }
    for(int i = 0; i < SecondModel::CTX_SIZE; i++) {
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
}

void SecondModel::displayState(const std::vector<std::string>& sentence) {
    std::array<int, SecondModel::CTX_SIZE> ctx;
    int answer;
    sentenceToContext(sentence, ctx, answer);
    printf("error negsample: %f\terror softmax:%f\twords: %d\n", errorNegSample(ctx, answer), errorSoftmax(ctx, answer), _vocabmgr->getVocabSize());
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

float SecondModel::errorNegSample(const std::array<int, SecondModel::CTX_SIZE>& ctx, int answer) {
    float result = error(ctx, answer, true);
    std::vector<int> sample = negSample(answer);
    for(int i : sample) {
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

Eigen::VectorXf SecondModel::derivW2(const std::array<int, SecondModel::CTX_SIZE>& ctx, int i, bool ok) {
    Eigen::VectorXf h = hidden(ctx);
    return (sigmoid(hypothesis(h, i))-ok)*h;
}

void SecondModel::gradCheck(const std::array<int, SecondModel::CTX_SIZE>& ctx, int j, int k) {
    float epsilon = sqrt(FLT_EPSILON);
    for(int b = 0; b < 2; ++b) {
        for(int i = 0; i < _n; ++i) {
            float tmp = _w1[ctx[j]](i);
            _w1[ctx[j]](i) = tmp + epsilon;
            float errpe = error(ctx, k, b);
            _w1[ctx[j]](i) = tmp - epsilon;
            float errme = error(ctx, k, b);
            _w1[ctx[j]](i) = tmp;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivW1(ctx, k, b)(i);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, i, b);
        }
        for(int i = 0; i < _n; ++i) {
            float tmp = _w2[k](i);
            _w2[k](i) = tmp + epsilon;
            float errpe = error(ctx, k, b);
            _w2[k](i) = tmp - epsilon;
            float errme = error(ctx, k, b);
            _w2[k](i) = tmp;
            float approx = (errpe-errme)/(2*epsilon);
            float calc = derivW2(ctx, k, b)(i);
            printf("\t%f\t%f\t%f\t%f\t%d\t%d\n", calc-approx, calc/approx, calc, approx, i, b);
        }
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
        if(nb == word) {
            --i;
            continue;
        }
        result.push_back(nb);
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

