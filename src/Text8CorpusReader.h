#ifndef TEXT8CORPUSREADER_H
#define TEXT8CORPUSREADER_H

#include "CorpusReader.h"
#include <fstream>

class Text8CorpusReader : public CorpusReader {
    public:
        Text8CorpusReader(const std::string& filename);
        ~Text8CorpusReader();
        virtual std::string readWord();
        virtual bool eof();
        virtual void startOver();
    protected:
        std::ifstream _file;
};

#endif

