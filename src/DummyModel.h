#ifndef DUMMYMODEL_H
#define DUMMYMODEL_H

#include "Model.h"

class DummyModel : public Model {
    virtual void train(const std::vector<std::string>& sentence, float alpha);
};

#endif

