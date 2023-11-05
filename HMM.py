import random
import argparse
import codecs
import os
import numpy as np


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n ' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


def load_helper(filename, dictionary):
    with codecs.open(filename, 'r', 'utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                if parts[0] not in dictionary:
                    dictionary[parts[0]] = {}
                dictionary[parts[0]][parts[1]] = float(parts[2])


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        load_helper(f"{basename}.trans", self.transitions)
        load_helper(f"{basename}.emit", self.emissions)

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """