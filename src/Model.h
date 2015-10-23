#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>

class Model {
    public:
        virtual void train(const std::vector<std::string>& sentence, float alpha) = 0;
        virtual int sentenceSize() = 0;
        virtual void save(const std::string& file) = 0;
        virtual void load(const std::string& file) = 0;
        virtual void displayState(const std::vector<std::string>& sentence) = 0;
};

#endif

