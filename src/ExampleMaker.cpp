#include "ExampleMaker.h"

ExampleMaker::ExampleMaker(CorpusReader* corpus, bool randomize) :
    _corpus(corpus), _randomize(randomize) {
}

std::vector<std::string> ExampleMaker::getExample(int n) {
    if(_randomize) {
        _corpus->randomizePosition();
    }
    std::vector<std::string> result;
    std::string current;
    for(int i = 0; i < n; ++i) {
        //If we are at the end of the file, start again
        if(_corpus->eof()) {
            _corpus->startOver();
            return getExample(n);
        }
        current = _corpus->readWord();
        if(keepWord(current)) {
            result.push_back(current);
        } else {
            --i;
        }
    }
    return result;
}

bool ExampleMaker::keepWord(const std::string& w) {
    return !w.empty();
}

