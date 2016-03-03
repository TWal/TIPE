#ifndef CORPUSREADER_H
#define CORPUSREADER_H

#include <string>
#include <vector>

class CorpusReader {
    public:
        virtual std::string readWord() = 0;
        virtual bool eof() = 0;
        virtual void startOver() = 0;
        virtual bool exampleBroken();
        virtual void clearBrokenFlag();
        virtual void randomizePosition() = 0;

    protected:
        bool _broken = false;
};

#endif

