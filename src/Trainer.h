#ifndef TRAINER_H
#define TRAINER_H

#include "Model.h"
#include "CorpusReader.h"

class Trainer {
    public:
        Trainer(Model* model, CorpusReader* corpus);
        void train();
        void test(int n);
        void infiniteTest(const std::string& filename);

    private:
        Model* _model;
        CorpusReader* _corpus;
};

#endif

