from util import load_dict
from copy import deepcopy
import numpy as np

class NgramMatrixFactory:
    """Assumes that it receives a batch of source files (and a vocab dictionary) and produces list of queries
    which is then sent to external LM for production (so that it populates a matrix with queries)"""
    def __init__(self, target_dict, ngram_order, n_words_target=-1):
        self.target_dict = load_dict(target_dict)
        self.ngram_order = ngram_order
        self.n_words_target = n_words_target

        # Due to not having proper BoS and EoS define them:
        self.BoS = 0
        self.EoS = 0
        if self.n_words_target != -1:
            self.BoS = self.n_words_target
            self.EoS = self.n_words_target + 1 #@TODO do we need EoS here (</s>)
        else:
            self.BoS = len(target_dict)
            self.EoS = len(target_dict) + 1 #@TODO do we need EoS here (</s>)

        # Limit the vocabulary if necessary.
        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.reverse_target_dict = {v: k for k, v in self.target_dict.iteritems()}
        self.reverse_target_dict[self.BoS] = "<s>"
        self.reverse_target_dict[self.EoS] = "</s>"

    ##Too slow, maybe NDarrays can speed it up but it seems to be wiser to just write out the ngrams
    ##and have gLM/kenlm produce the numbers and then load them
    def vocabSizeQueriesExpansion(self, queries):
        """Basically, replicate every single query target_vocab_size times and change the
        last word in the query to be from 0 to vocab_size"""
        vocabSizeQueries = []
        for query in queries:
            for i in range(self.n_words_target): #@TODO n_words_target + 1 mb, talk to rico
                vocabSizeQueries.append(deepcopy(query))
                vocabSizeQueries[-1][-1] = i
        return vocabSizeQueries
        

    def processBatch(self, target_sents):
        """Processes a batch of target sentences to form ngram queries"""
        queries = []
        for sentence in target_sents:
            for i in range(len(sentence)):
                query = []
                begin_num = i - self.ngram_order

                #Case 1: Having to Pad the beginning with BoS tokes <s>
                if begin_num < 0:
                    for j in range(-begin_num - 1):   
                        query.append(self.BoS)
                    sent_idx = 0
                    while len(query) < self.ngram_order:
                        query.append(sentence[sent_idx])
                        sent_idx += 1
                else:
                    #Case 2: just have a sliding window of size self.ngram_order until the end of sentences
                    query = sentence[begin_num:i]
                queries.append(query)
        return queries

    def writeToDisk(self, batch_queries, fileToWrite, reverse=False):
        ngrams_file = open(fileToWrite, "w")
        for query in batch_queries:
            if reverse: #E.G. gLM prefers reversed queries
                query.reverse()
            sent = ""
            for num in query:
                sent = sent + self.reverse_target_dict[num] + " "
            sent = sent[:-1] #remove trailing space
            ngrams_file.write(sent + '\n')
        ngrams_file.close()

    def dumpVocab(self, fileToWrite):
        """This dumps the vocabulary in the softmax order
        So far coded for gLM format
        @TODO this might not be the correct softmax order"""
        dictfile = open(fileToWrite, "w")
        dictfile.write(str(len(self.reverse_target_dict.keys())) + "\n")
        for key in self.reverse_target_dict:
            dictfile.write(str(key) + '\t' + self.reverse_target_dict[key] + "\n")
        dictfile.close()
