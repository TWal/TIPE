#!/bin/bash
mkdir data
cd data
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
rm text8.zip
wget https://word2vec.googlecode.com/svn/trunk/questions-words.txt
