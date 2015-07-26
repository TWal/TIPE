#include "Trainer.h"

Trainer::Trainer(Model* model, CorpusReader* corpus) :
    _model(model),
    _corpus(corpus) {
}

void Trainer::train() {
    _model->train(_corpus->readSentence(), 0.001);
}

