#ifndef TEXT8CORPUSREADER_H
#define TEXT8CORPUSREADER_H

#include "CorpusReader.h"
#include <fstream>
#include <random>

class Text8CorpusReader : public CorpusReader {
    public:
        Text8CorpusReader(const std::string& filename);
        ~Text8CorpusReader();
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

