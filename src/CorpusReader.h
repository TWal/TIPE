#ifndef CORPUSREADER_H
#define CORPUSREADER_H

#include <string>
#include <vector>

class CorpusReader {
    public:
        virtual std::string readWord() = 0;
        virtual bool eof() = 0;
        virtual void startOver() = 0;
        virtual void randomizePosition() = 0;
};

#endif

