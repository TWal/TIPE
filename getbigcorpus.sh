#!/bin/bash
normalize() {
        perl -pe 's/[^[:ascii:]]/#/g' | awk '{print tolower($0);}' | sed 's/[^a-z ]/ /g' | sed -re 's/\s+/ /g'
}

#Uses WikiExtractor.py, can be found at http://medialab.di.unipi.it/wiki/Wikipedia_Extractor

#English corpus (7B words)
mkdir corpus_en
cd corpus_en

wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
wget http://ebiquity.umbc.edu/redirect/to/resource/id/351/UMBC-webbase-corpus -Oumbc_webbase_corpus.tar.gz
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
for i in `seq 2007 2013`; do
       wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$i.en.shuffled.gz
done

python WikiExtractor.py -o wiki_extracted enwiki-latest-pages-articles.xml.bz2 --no-templates
tar -xvzf umbc_webbase_corpus.tar.gz
tar -xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
for i in `seq 2007 2013`; do
       gzip -d news.$i.en.shuffled.gz
done


find wiki_extracted -type f | xargs -d'\n' cat | perl -pe 's/<\/?doc.*?>//g' | normalize > corpus.txt;
ls -1 webbase_all/*.txt | xargs -d'\n' cat | normalize >> corpus.txt
ls -1 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* | xargs -d'\n' cat | normalize >> corpus.txt
ls -1 news.*.en.shuffled | xargs -d'\n' cat | normalize >> corpus.txt

cd ..

#French corpus (1.7B words)
mkdir corpus_fr
cd corpus_fr

wget http://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2
for i in `seq 2007 2013`; do
       wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.$i.fr.shuffled.gz
done

python WikiExtractor.py -o wiki_extracted_fr frwiki-latest-pages-articles.xml.bz2 --no-templates
for i in `seq 2007 2013`; do
       gzip -d news.$i.fr.shuffled.gz
done

find wiki_extracted_fr -type f | xargs -d'\n' cat | perl -pe 's/<\/?doc.*?>//g' | normalize > corpus.txt;
ls -1 news.*.fr.shuffled | xargs -d'\n' cat | normalize >> corpus.txt
