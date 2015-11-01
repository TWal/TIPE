#include "Trainer.h"

Trainer::Trainer(Model* model, ExampleMaker* ex) :
    _model(model),
    _ex(ex) {
}

void Trainer::train() {
    _model->train(_ex->getExample(_model->sentenceSize()), 0.025);
}

void Trainer::test(int n) {
    for (int i = 0; i <= n; i++) {
        train();
        if (i%100000 == 0) {
            _model->displayState(_ex->getExample(_model->sentenceSize()));
        }
    }
}

void Trainer::infiniteTest(const std::string& filename) {
    while(true) {
        for(int i = 0; i < 100000; ++i) {
            train();
        }
        _model->displayState(_ex->getExample(_model->sentenceSize()));
        _model->save(filename);
    }
}

