#include "LineCorpusReader.h"

LineCorpusReader::LineCorpusReader(const std::string& filename) {
    _file.open(filename.c_str());
    _file.seekg(0, _file.end);
    _dist = std::uniform_int_distribution<int>(0, int(_file.tellg())-1);
    _file.seekg(0);
}

LineCorpusReader::~LineCorpusReader() {
    _file.close();
}

std::string LineCorpusReader::readWord() {
    std::string result;
    std::getline(_file, result, ' ');
    //If we reach the end of file, remove the trailing \n
    if(_file.eof()) {
        return result.back() == '\n' ? result.substr(0, result.size()-1) : result;
    }
    //If a newline is reached, handle it
    size_t nl = result.find_first_of('\n');
    if(nl != std::string::npos) {
        _broken = true;
        _file.seekg(size_t(_file.tellg()) - result.size() + nl); //Go at the beginning of the line
        return result.substr(0, nl); //Return the last word of the last line
    }
    return result;
}

bool LineCorpusReader::eof() {
    return _file.eof();
}

void LineCorpusReader::startOver() {
    _file.clear(); //Remove the eof flag
    _file.seekg(0);
}

void LineCorpusReader::randomizePosition() {
    _file.seekg(_dist(_gen));
    readWord();
}
