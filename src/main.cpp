#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include "LineCorpusReader.h"
#include "SelectiveExampleMaker.h"
#include "DummyModel.h"
#include "SecondModel.h"
#include "Trainer.h"
#include "VocabManager.h"
#include "Distances.h"

void printUsage();

int main(int argc, char** argv) {
    if(std::string(argv[1]) == "train") {
        struct {
            int dim;
            int iter;
            std::string corpus;
            std::string output;
            std::string questions;
            bool random;
            int minCount;
            float alpha;
            float minAlpha;
            float downsample;
        } opts = {100, 5, "data/text8", "result.bin", "data/questions-words.txt", false, 5, 0.05, 0.0001, 1e-5};
        while(true) {
            static option longOptions[] = {
                {"dim", required_argument, 0, 'd'},
                {"iter", required_argument, 0, 'i'},
                {"corpus", required_argument, 0, 'c'},
                {"output", required_argument, 0, 'o'},
                {"questions", required_argument, 0, 'q'},
                {"random", no_argument, 0, 'r'},
                {"mincount", required_argument, 0, 'm'},
                {"alpha", required_argument, 0, 'a'},
                {"minalpha", required_argument, 0, 'b'},
                {"downsample", required_argument, 0, 's'}
            };
            int optionIndex = 0;
            int c = getopt_long(argc, argv, "d:i:c:o:q:rm:a:b:s:", longOptions, &optionIndex);
            if(c == -1) {
                break;
            }
            switch(c) {
                case 'd':
                    opts.dim = atoi(optarg);
                    break;
                case 'i':
                    opts.iter = atoi(optarg);
                    break;
                case 'c':
                    opts.corpus = optarg;
                    break;
                case 'o':
                    opts.output = optarg;
                    break;
                case 'q':
                    opts.questions = optarg;
                    break;
                case 'r':
                    opts.random = true;
                    break;
                case 'm':
                    opts.minCount = atoi(optarg);
                    break;
                case 'a':
                    opts.alpha = atof(optarg);
                    break;
                case 'b':
                    opts.minAlpha = atof(optarg);
                    break;
                case 's':
                    opts.downsample = atof(optarg);
                    break;
                default:
                    break;
            }
        }

        LineCorpusReader reader(opts.corpus);
        VocabManager vocabmgr;
        vocabmgr.compute(&reader);
        ExampleMaker* ex;
        if(opts.minCount > 0 || opts.downsample != 0) {
            ex = new SelectiveExampleMaker(&reader, &vocabmgr, opts.minCount, opts.downsample, opts.random);
        } else {
            ex = new ExampleMaker(&reader);
        }
        SecondModel model(opts.dim, &vocabmgr);
        Trainer trainer(&model, ex);
        trainer.train(vocabmgr.getTotal()*opts.iter, opts.alpha, opts.minAlpha);
        model.save(opts.output);
        model.checkAccuracy(opts.questions, Distances::cosinus);
    } else if(std::string(argv[1]) == "accuracy") {
        struct {
            std::string input;
            std::string questions;
        } opts = {"result.bin", "data/questions-words.txt"};
        while(true) {
            static option longOptions[] = {
                {"input", required_argument, 0, 'i'},
                {"questions", required_argument, 0, 'q'},
            };
            int optionIndex = 0;
            int c = getopt_long(argc, argv, "i:q:", longOptions, &optionIndex);
            if(c == -1) {
                break;
            }
            switch(c) {
                case 'i':
                    opts.input = optarg;
                    break;
                case 'q':
                    opts.questions = optarg;
                    break;
                default:
                    break;
            }
        }

        VocabManager vocabmgr;
        SecondModel model(&vocabmgr, opts.input);
        model.checkAccuracy(opts.questions, Distances::cosinus);
        return 0;
    } else {
        printUsage();
        return -1;
    }
}

void printUsage() {
    printf("Usage: todo\n");
}

