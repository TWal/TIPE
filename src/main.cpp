#include <stdio.h>
#include "Text8CorpusReader.h"
#include "SelectiveExampleMaker.h"
#include "DummyModel.h"
#include "SecondModel.h"
#include "Trainer.h"
#include "VocabManager.h"
#include "Distances.h"
#include <chrono>

int main() {
    if(true) {
        Text8CorpusReader reader("data/text8");
        VocabManager vocabmgr;
        vocabmgr.compute(&reader);
        SelectiveExampleMaker ex(&reader, &vocabmgr, 5, false);
        SecondModel model(100, &vocabmgr);
        Trainer trainer(&model, &ex);
        trainer.train(vocabmgr.getTotal()*1, 0.05, 0.0001);
        model.save("result.bin");
        model.checkAccuracy("data/questions-words.txt", Distances::cosinus);
    } else {
        VocabManager vocabmgr;
        SecondModel model(100, &vocabmgr);
        model.load("result.bin");
        model.checkAccuracy("data/questions-words.txt", Distances::cosinus);
    }
    return 0;
}
