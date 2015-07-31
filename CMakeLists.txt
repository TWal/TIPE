cmake_minimum_required(VERSION 2.6)

project(Tipe)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")
SET(CMAKE_CXX_FLAGS "-g -O2 -Wall -pedantic -std=c++11")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

set(${PROJECT_NAME}_SOURCES
    src/main.cpp
    src/Text8CorpusReader.cpp
    src/RandomText8CorpusReader.cpp
    src/DummyModel.cpp
    src/FirstModel.cpp
    src/Trainer.cpp
)

find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR})

add_executable(
    ${PROJECT_NAME}
    ${${PROJECT_NAME}_SOURCES}
)
