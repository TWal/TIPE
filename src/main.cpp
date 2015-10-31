#include <stdio.h>
#include "Text8CorpusReader.h"
#include "DummyModel.h"
#include "SecondModel.h"
#include "Trainer.h"

int main() {
    Text8CorpusReader reader("corpus/text8");
    SecondModel model(100, 253854);
    Trainer trainer(&model, &reader);
    trainer.infiniteTest("result.bin");
    return 0;
}
