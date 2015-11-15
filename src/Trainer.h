#ifndef TRAINER_H
#define TRAINER_H

#include "Model.h"
#include "ExampleMaker.h"

class Trainer {
    public:
        Trainer(Model* model, ExampleMaker* ex);
        void trainOnce(float alpha);
        void train(int n, float alpha);
        void train(int n, float alphaBegin, float alphaEnd);
        void infiniteTrain(float alpha, const std::string& filename);

    private:
        Model* _model;
        ExampleMaker* _ex;
};

#endif

