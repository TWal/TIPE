#ifndef SELECTIVEEXAMPLEMAKER_H
#define SELECTIVEEXAMPLEMAKER_H

#include "ExampleMaker.h"
#include "VocabManager.h"
#include <random>

class SelectiveExampleMaker : public ExampleMaker {
    public:
        SelectiveExampleMaker(CorpusReader* corpus, VocabManager* vocabmgr, int minCount, float downsample, bool randomize = false);
    private:
        virtual bool keepWord(const std::string& w);
        VocabManager* _vocabmgr;
        int _minCount;
        std::vector<float> _proba;
        std::mt19937 _gen;
        std::uniform_real_distribution<float> _dist;
};

#endif

