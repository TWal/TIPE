#include <stdio.h>
#include "Text8CorpusReader.h"
#include "DummyModel.h"
#include "FirstModel.h"
#include "Trainer.h"

int main() {
    Text8CorpusReader reader("corpus/text8");
    FirstModel model(25, 1);
    Trainer trainer(&model, &reader);
    trainer.infiniteTest("result.bin");
    return 0;
}
