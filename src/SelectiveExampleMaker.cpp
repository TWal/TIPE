#include "SelectiveExampleMaker.h"

SelectiveExampleMaker::SelectiveExampleMaker(CorpusReader* corpus, VocabManager* vocabmgr, int minCount, bool randomize) :
    ExampleMaker(corpus, randomize), _vocabmgr(vocabmgr), _minCount(minCount) {
    for(int i = _vocabmgr->getVocabSize()-1; i >= 0; --i) {
        if(_vocabmgr->getCount(i) < minCount) {
            _vocabmgr->removeWord(i, false);
        }
    }
    _vocabmgr->fixWordtoind();
}

bool SelectiveExampleMaker::keepWord(const std::string& w) {
    return !w.empty() && _vocabmgr->getIndex(w) != -1;
}

