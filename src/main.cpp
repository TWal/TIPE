#include <stdio.h>
#include "Text8CorpusReader.h"
#include "DummyModel.h"
#include "FirstModel.h"
#include "Trainer.h"

int main() {
    Text8CorpusReader reader("corpus/text8");
    DummyModel model;
    Trainer trainer(&model, &reader);
    FirstModel test(10, 1000);
    std::vector<int> phrase = {
        test.getWordInd("manger"),
        test.getWordInd("du"),
        test.getWordInd("chocolat"),
        test.getWordInd("noir"),
        test.getWordInd("miam"),
        test.getWordInd("c'est"),
        test.getWordInd("tres"),
        test.getWordInd("tres2"),
        test.getWordInd("tres3"),
        test.getWordInd("tres4"),
        test.getWordInd("tres5"),
        test.getWordInd("tres6"),
        test.getWordInd("bon")
    };

    test.gradCheck(phrase);


    /*
    for(int i = 0; i < 10; ++i) {
        trainer.train();
    }
    */

    return 0;
}
