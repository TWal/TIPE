#include "DummyModel.h"
#include <stdio.h>

void DummyModel::train(const std::vector<std::string>& sentence) {
    printf("Dummy training on: ");
    for(std::string s : sentence) {
        printf("%s ", s.c_str());
    }
    printf("\n");
}

