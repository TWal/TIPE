#ifndef TRAINER_H
#define TRAINER_H

#include "Model.h"
#include "CorpusReader.h"

class Trainer {
    public:
        Trainer(Model* model, CorpusReader* corpus);
        void train();
    private:
        Model* _model;
        CorpusReader* _corpus;
};

#endif

