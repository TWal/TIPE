#include "Serializer.h"

template<typename T>
static void writeT(T t, FILE* file) {
    fwrite(&t, sizeof(T), 1, file);
}

template<typename T>
static T readT(FILE* file) {
    T t;
    fread(&t, sizeof(T), 1, file);
    return t;
}

Serializer::Serializer() {

}

Serializer::~Serializer() {
    fclose(_file);
}

void Serializer::initRead(const std::string& filename) {
    _file = fopen(filename.c_str(), "rb");
}

void Serializer::initWrite(const std::string& filename) {
    _file = fopen(filename.c_str(), "wb");
}

void Serializer::writeFloat(float f) {
    writeT(f, _file);
}

float Serializer::readFloat() {
    return readT<float>(_file);
}

void Serializer::writeInt(int i) {
    writeT(i, _file);
}

int Serializer::readInt() {
    return readT<int>(_file);
}

void Serializer::writeUint64(uint64_t i) {
    writeT(i, _file);
}

uint64_t Serializer::readUint64() {
    return readT<uint64_t>(_file);
}

void Serializer::writeVec(const Eigen::VectorXf& v) {
    writeT(v.size(), _file);
    for(int i = 0; i < v.size(); ++i) {
        writeFloat(v(i));
    }
}

Eigen::VectorXf Serializer::readVec() {
    Eigen::VectorXf::Index size = readT<Eigen::VectorXf::Index>(_file);
    Eigen::VectorXf vec(size);
    for(int i = 0; i < size; ++i) {
        vec(i) = readFloat();
    }
    return vec;
}

void Serializer::writeMat(const Eigen::MatrixXf& m) {
    writeT(m.rows(), _file);
    writeT(m.cols(), _file);
    for(int i = 0; i < m.rows(); ++i) {
        for(int j = 0; j < m.cols(); ++j) {
            writeFloat(m(i, j));
        }
    }
}

Eigen::MatrixXf Serializer::readMat() {
    Eigen::MatrixXf::Index rows = readT<Eigen::MatrixXf::Index>(_file);
    Eigen::MatrixXf::Index cols = readT<Eigen::MatrixXf::Index>(_file);
    Eigen::MatrixXf m(rows, cols);
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            m(i, j) = readFloat();
        }
    }
    return m;
}

void Serializer::writeString(const std::string& s) {
    writeT(s.size(), _file);
    for(int i = 0; i < s.size(); ++i) {
        writeT(s[i], _file);
    }
}

std::string Serializer::readString() {
    std::string::size_type size = readT<std::string::size_type>(_file);
    std::string s;
    for(int i = 0; i < size; ++i) {
        s.push_back(readT<char>(_file));
    }
    return s;
}

