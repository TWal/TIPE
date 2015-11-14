#ifndef EXAMPLEMAKER_H
#define EXAMPLEMAKER_H

#include "CorpusReader.h"

class ExampleMaker {
    public:
        ExampleMaker(CorpusReader* corpus, bool randomize=false);
        virtual std::vector<std::string> getExample(int n);
        virtual std::vector<std::string> getLastExample();

    protected:
        virtual bool keepWord(const std::string& w);
        CorpusReader* _corpus;
        bool _randomize;
        std::vector<std::string> _currentExample;
};

#endif

