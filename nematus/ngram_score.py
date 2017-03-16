from util import load_dict
from copy import deepcopy
import numpy as np
import os, sys

class NgramMatrixFactory:
    """Assumes that it receives a batch of source files (and a vocab dictionary) and produces list of queries
    which is then sent to external LM for production (so that it populates a matrix with queries)"""
    def __init__(self, target_dict, ngram_order, n_words_target=-1):
        self.target_dict = load_dict(target_dict)
        self.ngram_order = ngram_order
        if n_words_target == -1:
            self.n_words_target = len(self.target_dict)
        else:
            self.n_words_target = n_words_target
        #Those vary with each batch sent
        self.sentence_length = 0
        self.batch_size = 0
        
        # Due to not having proper BoS and EoS define them:
        self.BoS = 0
        self.EoS = 0 # EoS needs to be 0 because that's how nematus uses it, BoS can be anything,
                     # we put it at the end of vocabulary
                     # @TODO this might depend on the dictionary used, but worry about that later.
        if self.n_words_target != -1:
            self.BoS = min(self.n_words_target, len(target_dict))
        else:
            self.BoS = len(target_dict)

        # Limit the vocabulary if necessary.
        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.reverse_target_dict = {v: k for k, v in self.target_dict.iteritems()}
        self.reverse_target_dict[self.BoS] = "<s>"
        self.reverse_target_dict[self.EoS] = "</s>"


    ##Too slow, maybe NDarrays can speed it up but it seems to be wiser to just write out the ngrams
    ##and have gLM/kenlm produce the numbers and then load them. It's kept only as a reference, don't use it.
    def vocabSizeQueriesExpansion(self, queries):
        """Basically, replicate every single query target_vocab_size times and change the
        last word in the query to be from 0 to vocab_size"""
        vocabSizeQueries = []
        for query in queries:
            for i in range(self.n_words_target): #@TODO n_words_target + 1 mb, talk to rico
                vocabSizeQueries.append(deepcopy(query))
                vocabSizeQueries[-1][-1] = i
        return vocabSizeQueries
        

    def sents2ngrams(self, target_sents):
        """Processes a batch of target sentences to form ngram queries
        The input is from the prepare_data fn of the nmt module"""
        sent_queries = []
        self.batch_size = len(target_sents.T)
        self.sentence_length = len(target_sents)
        print("Batch size: " + str(self.batch_size))
        print("Sentence sentence_length: " + str(self.sentence_length))
        for sentence in target_sents.T: #
            queries = []
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
            sent_queries.append(queries)
        return sent_queries

    #Nematus ngrams are expected to be ordered by word idx and not by sentence idx
    def writeToDisk(self, batch_queries, fileToWrite, reverse=False, wordIndexed=False):
        ngrams_file = open(fileToWrite, "w")

        if wordIndexed:
            for i in range(len(batch_queries[0])):
                for sentence in batch_queries:
                    query = sentence[i]
                    if reverse:
                        query = query[::-1]
                    sent = ""
                    for num in query:
                        sent = sent + self.reverse_target_dict[num] + " "
                    sent = sent[:-1] #remove trailing space
                    ngrams_file.write(sent + '\n')
        else:
            for sentence in batch_queries:
                for query in sentence:
                    if reverse: #E.G. gLM prefers reversed queries
                        query = query[::-1]
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

    def initGLM(self, path_to_gLM_module, path_to_LM, path_to_vocab, gpuMemoryUse, gpuDeviceID = 0):
        """This would init the gLM C object"""
        cmd_folder = os.path.realpath(os.path.abspath(path_to_gLM_module))
        if cmd_folder not in sys.path:
            sys.path.append(cmd_folder)
        import libngrams_nematus
        self.gLM = libngrams_nematus.NematusLM(path_to_LM, path_to_vocab, gpuMemoryUse, gpuDeviceID)

    def getScoresForBatch(self, target_sents, tmp_file):
        """Given a list of target sentences, returns an ndarray of all the queries"""
        ngrams_batch = self.sents2ngrams(target_sents)
        self.writeToDisk(ngrams_batch, tmp_file, True, True)
        return self.gLM.processBatch(tmp_file, self.n_words_target, self.sentence_length, self.batch_size)

    #This is how we clear the cMemory taken by all existing ndarrays
    def clearMemory(self):
        """Calls a C function to clear the memory used from all arrays thus far"""
        self.gLM.freeResultsMemory()

if __name__ == '__main__':
    #Test
    from data_iterator import TextIterator
    from nmt import prepare_data
    a = TextIterator("../../de_en_wmt16/dev.bpe.de", "../../de_en_wmt16/dev.bpe.en",\
     ["../../de_en_wmt16/vocab.de.pkl"], "../../de_en_wmt16/vocab.en.pkl", 128, 100, -1, 30000)
    source, target = a.next()
    source_padded, source_mask, target_padded, target_mask, _ = prepare_data(source, target)
    ngrams = NgramMatrixFactory("../../de_en_wmt16/vocab.en.pkl", 6, 30000)
    ngrams.dumpVocab("/tmp/dictfile") 
    ngrams.initGLM('/home/dheart/uni_stuff/phd_2/gLM/release_build/lib', \
        '/home/dheart/uni_stuff/phd_2/dl4mt-tutorial/de_en_wmt16/bpe_sents_4.glm/',
         '/tmp/dictfile', 2900, 0)
    scores = ngrams.getScoresForBatch(target_padded, '/tmp/tmpngrams')
    #Don't forget to clear memory after use!


    
