#include "FirstModel.h"

static float carre(float x) {
	return x*x;
}

void FirstModel::train(const std::vector<std::string>& sentence) {
}

float FirstModel::hypothesis(int j, std::vector<int> phrase) {
	float S = 0;
	for (int l = 0; l < 4*n; l++) {
		float valthet = theta[l][j];
		int c = l/n;
		if (c>=2) {
			c++;
		}
		int k = l%n;
		float valvec = indtovec[phrase[c]][k];
		S += valthet*valvec;
	}
	return S;
}

float FirstModel::error(std::vector<int> phrase) {
	float S = 0;
	for (int j = 0; j<n; j++) {
		float x = hypothesis(j,phrase) - indtovec[phrase[2]][j];
		S += carre(x);
		for (int l = 0; l<4*n; l++) {
			S += carre(theta[l][j]);
		}
		for (int i = 0; i<5; i++) {
			S += carre(indtovec[phrase[i]][j]);
		}
	}
	return S/2;
}

float FirstModel::derivtheta (int l, int j, std::vector<int> phrase) {
	int c = l/n;
	c += (c>=2);
	int k = l%n;
	return (hypothesis(j, phrase)-indtovec[phrase[2]][j])*indtovec[phrase[c]][k];
}

float FirstModel::derivmot (int c, int k, std::vector<int> phrase) {
	if (c == 2) {
		return indtovec[phrase[c]][k]-hypothesis(k, phrase);
	}
	else {
		float S = 0;
		for (int j=0; j < n; j++) {
			S += theta[(c-(c>2))*n+k][j]*(hypothesis(j,phrase)-indtovec[phrase[2]][j]);
		}
		return S;
	}
}

