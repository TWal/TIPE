#include "FirstModel.h"
#include <random>

//Magic, untested values
static const float RANDOM_MEAN = 0.0;
static const float RANDOM_STDDEV = 3.0;

static float carre(float x) {
	return x*x;
}

FirstModel::FirstModel(int n) :
    _n(n),
    _theta(4*n, std::vector<float>(n, 0.0)) {

    std::mt19937 gen;
    std::normal_distribution<float> dist(RANDOM_MEAN, RANDOM_STDDEV);
    for(int i = 0; i < 4*n; ++i) {
        for(int j = 0; j < n; ++j) {
            _theta[i][j] = dist(gen);
        }
    }
}

void FirstModel::train(const std::vector<std::string>& sentence) {
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
		float x = hypothesis(j,phrase) - _indtovec[phrase[2]][j];
		S += carre(x);
		for (int l = 0; l<4*_n; l++) {
			S += carre(_theta[l][j]);
		}
		for (int i = 0; i<5; i++) {
			S += carre(_indtovec[phrase[i]][j]);
		}
	}
	return S/2;
}

float FirstModel::derivTheta(int l, int j, const std::vector<int>& phrase) {
	int c = l/_n;
	c += (c>=2);
	int k = l%_n;
	return (hypothesis(j, phrase)-_indtovec[phrase[2]][j])*_indtovec[phrase[c]][k];
}

float FirstModel::derivWord(int c, int k, const std::vector<int>& phrase) {
	if (c == 2) {
		return _indtovec[phrase[c]][k]-hypothesis(k, phrase);
	} else {
		float S = 0;
		for (int j=0; j < _n; j++) {
			S += _theta[(c-(c>2))*_n+k][j]*(hypothesis(j,phrase)-_indtovec[phrase[2]][j]);
		}
		return S;
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

