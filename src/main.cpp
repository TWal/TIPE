#include <stdio.h>
#include "Text8CorpusReader.h"
#include "DummyModel.h"
#include "FirstModel.h"
#include "Trainer.h"

int main() {
    Text8CorpusReader reader("corpus/text8");
    FirstModel model(10, 1);
    Trainer trainer(&model, &reader);
    trainer.test(10000);
    return 0;
}
