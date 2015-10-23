#include "Trainer.h"
#include "FirstModel.h"
#include "CorpusReader.h"
#include <algorithm>

Trainer::Trainer(Model* model, CorpusReader* corpus) :
    _model(model),
    _corpus(corpus) {
}

void Trainer::train() {
    _model->train(_corpus->readSentence(_model->sentenceSize()), 0.001);
}

void Trainer::test(int n) {
    for (int i = 0; i <= n; i++) {
        train();
        if (i%1000 == 0) {
            _model->displayState(_corpus->readSentence(_model->sentenceSize()));
        }
    }
}

void Trainer::infiniteTest(const std::string& filename) {
    while(true) {
        for(int i = 0; i < 10000; ++i) {
            train();
        }
        _model->displayState(_corpus->readSentence(_model->sentenceSize()));
        _model->save(filename);
    }
}

