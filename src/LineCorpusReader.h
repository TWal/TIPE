#ifndef LINECORPUSREADER_H
#define LINECORPUSREADER_H

#include "CorpusReader.h"
#include <fstream>
#include <random>

class LineCorpusReader : public CorpusReader {
    public:
        LineCorpusReader(const std::string& filename);
        ~LineCorpusReader();
        virtual std::string readWord();
        virtual bool eof();
        virtual void startOver();
        virtual void randomizePosition();

    protected:
        std::ifstream _file;
        std::mt19937 _gen;
        std::uniform_int_distribution<int> _dist;
};

#endif

