#include "RandomText8CorpusReader.h"

RandomText8CorpusReader::RandomText8CorpusReader(const std::string& filename) :
    Text8CorpusReader(filename) {

    _file.seekg(0, _file.end);
    _dist = std::uniform_int_distribution<int>(0, int(_file.tellg())-1);
    _file.seekg(0);
}

std::vector<std::string> RandomText8CorpusReader::readSentence(int n) {
    _file.seekg(_dist(_gen));
    readWord();
    return Text8CorpusReader::readSentence(n);
}

