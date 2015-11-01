#ifndef EXAMPLEMAKER_H
#define EXAMPLEMAKER_H

#include "CorpusReader.h"

class ExampleMaker {
    public:
        ExampleMaker(CorpusReader* corpus, bool randomize=false);
        virtual std::vector<std::string> getExample(int n);

    protected:
        virtual bool keepWord(const std::string& w);
        CorpusReader* _corpus;
        bool _randomize;
};

#endif

