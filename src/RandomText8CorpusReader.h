#ifndef RANDOMTEXT8CORPUSREADER_H
#define RANDOMTEXT8CORPUSREADER_H

#include "Text8CorpusReader.h"
#include <random>

class RandomText8CorpusReader : public Text8CorpusReader {
    public:
        RandomText8CorpusReader(const std::string& filename);
        virtual std::vector<std::string> readSentence(int n);
    protected:
        std::mt19937 _gen;
        std::uniform_int_distribution<int> _dist;
};

#endif

