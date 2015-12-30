#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <Eigen/Dense>
#include <cstdio>

class Serializer {
    public:
        Serializer();
        ~Serializer();
        void initRead(const std::string& filename);
        void initWrite(const std::string& filename);

        void writeFloat(float f);
        void writeInt(int i);
        void writeUint64(uint64_t i);
        void writeVec(const Eigen::VectorXf& v);
        void writeMat(const Eigen::MatrixXf& m);
        void writeString(const std::string& s);

        float readFloat();
        int readInt();
        uint64_t readUint64();
        Eigen::VectorXf readVec();
        Eigen::MatrixXf readMat();
        std::string readString();

    private:
        FILE* _file;
};

#endif

