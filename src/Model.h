#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>

class Model {
    public:
        virtual void train(const std::vector<std::string>& sentence) = 0;
};

#endif

