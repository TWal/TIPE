#include "Text8CorpusReader.h"

Text8CorpusReader::Text8CorpusReader(const std::string& filename) {
    _file.open(filename.c_str());
}

Text8CorpusReader::~Text8CorpusReader() {
    _file.close();
}

std::vector<std::string> Text8CorpusReader::readSentence(int n) {
    //Suppose we are at the beginning of a word, read n words
    std::vector<std::string> result;
    std::string current;
    for(int i = 0; i < n; ++i) {
        //If we are at the end of the file, start again
        if(_file.eof()) {
            _file.clear();
            _file.seekg(0);
            return readSentence(n);
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

