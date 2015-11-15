#include "Trainer.h"

Trainer::Trainer(Model* model, ExampleMaker* ex) :
    _model(model),
    _ex(ex) {
}

void Trainer::trainOnce(float alpha) {
    _model->train(_ex->getExample(_model->sentenceSize()), alpha);
}

void Trainer::train(int n, float alpha) {
    for (int i = 0; i <= n; i++) {
        trainOnce(alpha);
        if (i%100000 == 0) {
            _model->displayState(_ex->getLastExample());
        }
    }
}

void Trainer::train(int n, float alphaBegin, float alphaMin) {
    for(int i = 0; i < n; ++i) {
        float alpha = std::max(alphaBegin*(1-float(i)/n), alphaMin);
        trainOnce(alpha);
        if(i%100000 == 0) {
            printf("Progress: %f\tAlpha: %f\n", 100*float(i)/n, alpha);
            _model->displayState(_ex->getLastExample());
        }
    }
}

void Trainer::infiniteTrain(float alpha, const std::string& filename) {
    while(true) {
        for(int i = 0; i < 100000; ++i) {
            trainOnce(alpha);
        }
        _model->displayState(_ex->getLastExample());
        _model->save(filename);
    }
}

