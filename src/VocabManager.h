#ifndef VOCABMANAGER_H
#define VOCABMANAGER_H

#include <unordered_map>
#include <vector>
#include <string>
#include "CorpusReader.h"
#include "Serializer.h"

class VocabManager {
    public:
        void compute(CorpusReader* corpus);
        int getCount(int ind);
        int getIndex(const std::string& word);
        const std::string& getWord(int ind);
        uint64_t getTotal();
        int getVocabSize();
        void save(Serializer& s);
        void load(Serializer& s);
        void removeCountBelow(int minCount);

    private:
        std::unordered_map<std::string, int> _wordtoind;
        std::vector<std::string> _indtoword;
        std::vector<uint64_t> _counts;
        uint64_t _total;
};

#endif

