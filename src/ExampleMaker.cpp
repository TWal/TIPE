#include "ExampleMaker.h"
#include <cassert>

ExampleMaker::ExampleMaker(CorpusReader* corpus, bool randomize) :
    _corpus(corpus), _randomize(randomize) {
}

std::vector<std::string> ExampleMaker::getExample(int n) {
    if(_randomize) {
        _corpus->randomizePosition();
        _currentExample.clear();
    }

    int newWords;
    if(_currentExample.empty()) {
        newWords = n;
    } else {
        newWords = 1;
        assert(n == _currentExample.size()); //To make the code simpler
        _currentExample.erase(_currentExample.begin());
    }

    std::string current;
    for(int i = 0; i < newWords; ++i) {
        //If we are at the end of the file, start again
        if(_corpus->eof()) {
            _corpus->startOver();
            _currentExample.clear();
            return getExample(n);
        }
        if(_corpus->exampleBroken()) {
            _currentExample.clear();
            _corpus->clearBrokenFlag();
            return getExample(n);
        }
        current = _corpus->readWord();
        if(keepWord(current)) {
            _currentExample.push_back(current);
        } else {
            --i;
        }
    }
    return _currentExample;
}

std::vector<std::string> ExampleMaker::getLastExample() {
    return _currentExample;
}

bool ExampleMaker::keepWord(const std::string& w) {
    return !w.empty();
}

