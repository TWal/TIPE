#ifndef SELECTIVEEXAMPLEMAKER_H
#define SELECTIVEEXAMPLEMAKER_H

#include "ExampleMaker.h"
#include "VocabManager.h"

class SelectiveExampleMaker : public ExampleMaker {
    public:
        SelectiveExampleMaker(CorpusReader* corpus, VocabManager* vocabmgr, int minCount, bool randomize = false);
    private:
        virtual bool keepWord(const std::string& w);
        VocabManager* _vocabmgr;
        int _minCount;
};

#endif

