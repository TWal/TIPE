#include "Trainer.h"

#include <chrono>
#include <cmath>

Trainer::Trainer(Model* model, ExampleMaker* ex) :
    _model(model),
    _ex(ex) {
}

void Trainer::trainOnce(float alpha) {
    _model->train(_ex->getExample(_model->sentenceSize()), alpha);
}

void Trainer::train(uint64_t n, float alpha) {
    for (int i = 0; i <= n; i++) {
        trainOnce(alpha);
        if (i%100000 == 0) {
            _model->displayState(_ex->getLastExample());
        }
    }
}

void Trainer::train(uint64_t n, float alphaBegin, float alphaMin) {
    std::chrono::high_resolution_clock::time_point last = std::chrono::high_resolution_clock::now();
    const int messageDistance = 100000;
    for(uint64_t i = 1; i <= n; ++i) {
        float alpha = std::max(alphaBegin*(1-float(i)/n), alphaMin);
        trainOnce(alpha);
        if(i%messageDistance == 0) {
            std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(now-last).count();
            double remaining = duration/messageDistance*(n-i);
            printf("\rProgress: %.2f%%\tAlpha: %.4f\tWords/s: %.2fk\tRemaining: %dh%.1fmin\t|\t", 100*float(i)/n, alpha, messageDistance/duration/1000, (int)(remaining/3600), fmod(remaining, 3600)/60);
            last = now;
            _model->displayState(_ex->getLastExample());
            fflush(stdout);
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

