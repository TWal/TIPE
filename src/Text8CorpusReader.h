#ifndef TEXT8CORPUSREADER_H
#define TEXT8CORPUSREADER_H

#include "CorpusReader.h"
#include <fstream>

class Text8CorpusReader : public CorpusReader {
    public:
        Text8CorpusReader(const std::string& filename);
        ~Text8CorpusReader();
        virtual std::vector<std::string> readSentence(int n);
    protected:
        std::ifstream _file;
};

#endif

