#ifndef DUMMYMODEL_H
#define DUMMYMODEL_H

#include "Model.h"

class DummyModel : public Model {
    public:
        virtual void train(const std::vector<std::string>& sentence, float alpha);
        virtual float error(const std::vector<std::string>& sentence);
        virtual int vocabSize();
};

#endif

