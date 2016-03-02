#include "SelectiveExampleMaker.h"

SelectiveExampleMaker::SelectiveExampleMaker(CorpusReader* corpus, VocabManager* vocabmgr, int minCount, float downsample, bool randomize) :
    ExampleMaker(corpus, randomize), _vocabmgr(vocabmgr), _minCount(minCount), _dist(0,1) {
    _vocabmgr->removeCountBelow(minCount);
    for (int i = 0; i < _vocabmgr->getVocabSize(); ++i) {
        _proba.push_back(sqrt((_vocabmgr->getTotal())*downsample/(_vocabmgr->getCount(i))));
    }
 }
bool SelectiveExampleMaker::keepWord(const std::string& w) {
    int id = _vocabmgr->getIndex(w);
    return !w.empty() && id != -1 && (_proba[id] >= 1 || _dist(_gen) < _proba[id]);
}
