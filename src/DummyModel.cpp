#include "DummyModel.h"
#include <stdio.h>

void DummyModel::train(const std::vector<std::string>& sentence, float alpha) {
    printf("Dummy training on: ");
    for(std::string s : sentence) {
        printf("%s ", s.c_str());
    }
    printf("\n");
}

float DummyModel::error(const std::vector<std::string>& sentence) {
    return -1;
}

int DummyModel::vocabSize() {
    return -1;
}

