#include <stdio.h>
#include "Text8CorpusReader.h"
#include "SelectiveExampleMaker.h"
#include "DummyModel.h"
#include "SecondModel.h"
#include "Trainer.h"
#include "VocabManager.h"

int main() {
    Text8CorpusReader reader("corpus/text8");
    VocabManager vocabmgr;
    vocabmgr.compute(&reader);
    SelectiveExampleMaker ex(&reader, &vocabmgr, 5);
    SecondModel model(100, &vocabmgr);
    Trainer trainer(&model, &ex);
    trainer.infiniteTest("result.bin");
    return 0;
}
