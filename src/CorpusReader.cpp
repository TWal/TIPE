#include "CorpusReader.h"

std::vector<std::string> CorpusReader::readSentence(int n) {
    //Suppose we are at the beginning of a word, read n words
    std::vector<std::string> result;
    std::string current;
    for(int i = 0; i < n; ++i) {
        //If we are at the end of the file, start again
        if(eof()) {
            startOver();
            return readSentence(n);
        }
        current = readWord();
        // "" is not a word
        if(!current.empty()) {
            result.push_back(current);
        } else {
            --i;
        }
    }

    return result;
}
