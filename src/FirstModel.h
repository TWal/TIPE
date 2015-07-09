#ifndef FIRSTMODEL_H
#define FIRSTMODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include "Model.h"

class FirstModel : public Model {
	public:
		virtual void train(const std::vector<std::string>& sentence);
		float hypothesis(int j, std::vector<int> phrase);
		float error(std::vector<int> phrase);
		float derivtheta (int l, int j, std::vector<int> phrase);
		float derivmot (int c, int k, std::vector<int> phrase);
	private:
		std::vector<std::vector<float>> theta;
		std::vector<std::vector<float>> indtovec;
		int n;
		std::unordered_map<std::string,int> wordtoind;
};

#endif

