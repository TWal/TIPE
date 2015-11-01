#include <stdio.h>
#include "Text8CorpusReader.h"
#include "ExampleMaker.h"
#include "DummyModel.h"
#include "SecondModel.h"
#include "Trainer.h"

int main() {
    Text8CorpusReader reader("corpus/text8");
    ExampleMaker ex(&reader);
    SecondModel model(100, 253854);
    Trainer trainer(&model, &ex);
    trainer.infiniteTest("result.bin");
    return 0;
}
