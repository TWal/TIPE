#include <stdio.h>
#include "Text8CorpusReader.h"
#include "DummyModel.h"
#include "Trainer.h"

int main() {
    Text8CorpusReader reader("corpus/text8");
    DummyModel model;
    Trainer trainer(&model, &reader);

    for(int i = 0; i < 10; ++i) {
        trainer.train();
    }

    return 0;
}
