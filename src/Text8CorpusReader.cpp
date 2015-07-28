#include "Text8CorpusReader.h"

Text8CorpusReader::Text8CorpusReader(const std::string& filename) {
    _file.open(filename.c_str());
}

Text8CorpusReader::~Text8CorpusReader() {
    _file.close();
}

std::vector<std::string> Text8CorpusReader::readSentence() {
    //Suppose we are at the beginning of a word, read 20 words
    std::vector<std::string> result;
    std::string current;
    for(int i = 0; i < 20; ++i) {
        //If we are at the end of the file, start again
        if(_file.eof()) {
            _file.clear();
            _file.seekg(0);
            return readSentence();
        }
        std::getline(_file, current, ' ');
        // "" is not a word
        if(!current.empty()) {
            result.push_back(current);
        } else {
            --i;
        }
    }

    return result;
}

