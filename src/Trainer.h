#ifndef TRAINER_H
#define TRAINER_H

#include "Model.h"
#include "ExampleMaker.h"

class Trainer {
    public:
        Trainer(Model* model, ExampleMaker* ex);
        void train();
        void test(int n);
        void infiniteTest(const std::string& filename);

    private:
        Model* _model;
        ExampleMaker* _ex;
};

#endif

