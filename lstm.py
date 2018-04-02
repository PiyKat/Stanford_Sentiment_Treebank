import numpy as np
import tensorflow as tf
import pandas as pd
import dill as pickle

from collections import defaultdict

import os


class Word2Vec():

    def __init__(self):

        self.word2vec_dim = 300
        default_arr = lambda: np.zeros(shape=[1, self.word2vec_dim])

        self.word_to_vector = defaultdict(default_arr)
        self.create_flag = 0


    def create_glove_vectors(self, path):

        if self.create_flag == 0:

            with open(path, "r") as f:

                for line in f.readlines():
                    newline_split = line.split()

                    self.word_to_vector[newline_split[0]] = np.array(list(map(np.float32, newline_split[1:])))

                    print("Word vector for " + str(newline_split[0] + " created"))

            print("Word vector dictionary created !!!!!!!")

            with open("./w2v_dic.pkl","wb") as w:

                pickle.dump(self.word_to_vector,w)

            self.create_flag = 1

        else:

            print("Word vectors already created !!!!!")
            self.load_word2vec()

    def load_word2vec(self):

        if os.path.exists("./w2v_dic.pkl"):

            with open("./w2v_dic.pkl","rb") as f:

                self.word_to_vector = pickle.load(f)

        else:

            print("Word Vector not created. Please call create_word2vec first.")


    def get_word2vec(self, word):

        return self.word_to_vector[word]


    def check_word(self, word):

        if word in self.word_to_vector.keys():

            return True

        else:

            return False

    def get_word2vec_seq(self, word_seq):

        #### We will do padding using tf.pad #####

        word_sequence = word_seq.split()

        sentence_arr = []

        for word in word_sequence:

            if word_sequence.index(word) == 0:

                sentence_arr = self.get_word2vec(word)

            else:

                word_arr = self.get_word2vec(word)
                sentence_arr = np.append(sentence_arr, word_arr, axis=0)

        return sentence_arr


