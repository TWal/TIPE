#include "Trainer.h"
#include "FirstModel.h"
#include "CorpusReader.h"
#include <algorithm>

Trainer::Trainer(Model* model, CorpusReader* corpus) :
    _model(model),
    _corpus(corpus) {
}

void Trainer::train() {
    _model->train(_corpus->readSentence(), 0.001);
}

void Trainer::test(int n) {
    for (int i = 0; i <= n; i++) {
        train();
        if (i%1000 == 0) {
            float e = _model->error(_corpus->readSentence());
            printf("%d\t error: %f\twords: %d\n", i, e, _model->vocabSize());
        }
    }
}

void Trainer::infiniteTest(const std::string& filename) {
    while(true) {
        for(int i = 0; i < 10000; ++i) {
            train();
        }
        float e = _model->error(_corpus->readSentence());
        printf("error: %f\twords: %d\n", e, _model->vocabSize());
        _model->save(filename);
    }
}

