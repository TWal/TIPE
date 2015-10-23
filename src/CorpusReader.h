#ifndef CORPUSREADER_H
#define CORPUSREADER_H

#include <string>
#include <vector>

class CorpusReader {
    public:
        virtual std::vector<std::string> readSentence(int n) = 0;
};

#endif

