#include "Text8CorpusReader.h"

Text8CorpusReader::Text8CorpusReader(const std::string& filename) {
    _file.open(filename.c_str());
    _file.seekg(0, _file.end);
    _dist = std::uniform_int_distribution<int>(0, int(_file.tellg())-1);
    _file.seekg(0);
}

Text8CorpusReader::~Text8CorpusReader() {
    _file.close();
}

std::string Text8CorpusReader::readWord() {
    std::string result;
    std::getline(_file, result, ' ');
    return result;
}

bool Text8CorpusReader::eof() {
    return _file.eof();
}

void Text8CorpusReader::startOver() {
    _file.clear();
    _file.seekg(0);
}

void Text8CorpusReader::randomizePosition() {
    _file.seekg(_dist(_gen));
    readWord();
}
