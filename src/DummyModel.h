#ifndef DUMMYMODEL_H
#define DUMMYMODEL_H

#include "Model.h"

class DummyModel : public Model {
    public:
        virtual void train(const std::vector<std::string>& sentence, float alpha);
        virtual int sentenceSize();
        virtual void save(const std::string& file);
        virtual void load(const std::string& file);
        virtual void displayState(const std::vector<std::string>& sentence);
};

#endif

