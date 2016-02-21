#include "SelectiveExampleMaker.h"

SelectiveExampleMaker::SelectiveExampleMaker(CorpusReader* corpus, VocabManager* vocabmgr, int minCount, bool randomize) :
    ExampleMaker(corpus, randomize), _vocabmgr(vocabmgr), _minCount(minCount) {
    _vocabmgr->removeCountBelow(minCount);
}

bool SelectiveExampleMaker::keepWord(const std::string& w) {
    return !w.empty() && _vocabmgr->getIndex(w) != -1;
}

