#include "VocabManager.h"

void VocabManager::compute(CorpusReader* corpus) {
    corpus->startOver();
    _total = 0;
    while (!corpus->eof()) {
        std::string word = corpus->readWord();
        if(word.empty()) {
            continue;
        }
        auto it = _wordtoind.find(word);
        if (it == _wordtoind.end()) {
            int ind = _wordtoind.size();
            _indtoword.push_back(word);
            _wordtoind.emplace(word, ind);
            _counts.push_back(1);
        }
        else {
            _counts[it->second]++;
        }
        _total++;
    }
}

int VocabManager::getCount(int ind) {
    return _counts[ind];
}

int VocabManager::getIndex(const std::string& word) {
    auto it = _wordtoind.find(word);
    return it == _wordtoind.end() ? -1 : _wordtoind[word];
}

const std::string& VocabManager::getWord(int ind) {
    return _indtoword[ind];
}

uint64_t VocabManager::getTotal() {
    return _total;
}

int VocabManager::getVocabSize() {
    return _indtoword.size();
}

void VocabManager::save(Serializer& s) {
    s.writeInt(_wordtoind.size());
    for(const std::pair<std::string, int>& p : _wordtoind) {
        s.writeString(p.first);
        s.writeInt(p.second);
    }
    for(const std::string& str : _indtoword) {
        s.writeString(str);
    }
    for(int count : _counts) {
        s.writeUint64(count);
    }
    s.writeUint64(_total);
}

void VocabManager::load(Serializer& s) {
    int size = s.readInt();

    _wordtoind.clear();
    for(int i = 0; i < size; ++i) {
        std::string word = s.readString();
        int ind = s.readInt();
        _wordtoind.emplace(word, ind);
    }

    _indtoword.clear();
    for(int i = 0; i < size; ++i) {
        _indtoword.push_back(s.readString());
    }
    _counts.clear();
    for(int i = 0; i < size; ++i) {
        _counts.push_back(s.readUint64());
    }
    _total = s.readUint64();
}

void VocabManager::removeWord(int ind, bool fixWordtoind) {
    _total -= _counts[ind];
    _wordtoind.erase(_wordtoind.find(_indtoword[ind]));
    _counts.erase(_counts.begin()+ind);
    _indtoword.erase(_indtoword.begin()+ind);
    if(fixWordtoind) {
        for(int i = ind+1; i < _indtoword.size(); ++i) {
            _wordtoind[_indtoword[i]] -= 1;
        }
    }
}

void VocabManager::fixWordtoind() {
    for(int i = 0; i < _indtoword.size(); ++i) {
        _wordtoind[_indtoword[i]] = i;
    }
}

