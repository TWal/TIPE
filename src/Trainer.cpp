#include "Trainer.h"

#include <chrono>

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
    std::chrono::high_resolution_clock::time_point last = std::chrono::high_resolution_clock::now();
    const int messageDistance = 100000;
    for(int i = 1; i <= n; ++i) {
        float alpha = std::max(alphaBegin*(1-float(i)/n), alphaMin);
        trainOnce(alpha);
        if(i%messageDistance == 0) {
            std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            printf("Progress: %.2f%%\tAlpha: %.4f\tWords/s: %.2fk\n", 100*float(i)/n, alpha, messageDistance/std::chrono::duration_cast<std::chrono::duration<double>>(now-last).count()/1000);
            last = now;
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

