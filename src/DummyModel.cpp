#include "DummyModel.h"
#include <stdio.h>

void DummyModel::train(const std::vector<std::string>& sentence, float alpha) {
    printf("Dummy training on: ");
    for(std::string s : sentence) {
        printf("%s ", s.c_str());
    }
    printf("\n");
}

int DummyModel::sentenceSize() {
    return 5;
}

void DummyModel::save(const std::string& file) {
}

void DummyModel::load(const std::string& file) {
}

void DummyModel::displayState(const std::vector<std::string>& sentence) {
    printf("Dummy::displayState called\n");
}

