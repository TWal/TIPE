#include <stdio.h>
#include "Text8CorpusReader.h"
#include "SelectiveExampleMaker.h"
#include "DummyModel.h"
#include "SecondModel.h"
#include "Trainer.h"
#include "VocabManager.h"
#include "Distances.h"

int main() {
    if(true) {
        Text8CorpusReader reader("data/text8");
        VocabManager vocabmgr;
        vocabmgr.compute(&reader);
        SelectiveExampleMaker ex(&reader, &vocabmgr, 5, true);
        SecondModel model(100, &vocabmgr);
        Trainer trainer(&model, &ex);
        trainer.infiniteTest("result.bin");
    } else {
        VocabManager vocabmgr;
        SecondModel model(100, &vocabmgr);
        model.load("result.bin");
        model.checkAccuracy("data/questions-words.txt", Distances::cosinus, true);
    }
    return 0;
}
