cmake_minimum_required(VERSION 2.6)

project(Tipe)
SET(CMAKE_CXX_FLAGS "-g -O2 -Wall -pedantic -std=c++11")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

set(${PROJECT_NAME}_SOURCES
    src/main.cpp
    src/Text8CorpusReader.cpp
    src/DummyModel.cpp
    src/Trainer.cpp
)

add_executable(
    ${PROJECT_NAME}
    ${${PROJECT_NAME}_SOURCES}
)
