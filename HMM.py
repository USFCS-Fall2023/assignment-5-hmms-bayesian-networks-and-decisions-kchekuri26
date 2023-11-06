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
        states = ['#']
        emissions = []

        for _ in range(n):
            current_state = states[-1]
            next_state = np.random.choice(list(self.transitions[current_state].keys()), p=list(self.transitions[current_state].values()))
            states.append(next_state)

            emission = np.random.choice(list(self.emissions[next_state].keys()), p=list(self.emissions[next_state].values()))
            emissions.append(emission)

        return Observation(states[1:], emissions)

    def forward(self, observation):
        N = len(observation.outputseq)
        states = list(self.transitions.keys())
        alpha = {state: [0] * N for state in states}

        for state in states:
            if state != '#':
                alpha[state][0] = self.transitions['#'].get(state, 0) * self.emissions[state].get(observation.outputseq[0], 0)

        for n in range(1, N):
            for next_state in states:
                if next_state != '#':
                    sum_alpha = sum(alpha[curr_state][n - 1] * self.transitions[curr_state].get(next_state, 0) for curr_state in states if curr_state != '#')
                    alpha[next_state][n] = sum_alpha * self.emissions[next_state].get(observation.outputseq[n], 0)

        final_probs = {state: alpha[state][-1] for state in states if state != '#'}
        final_state = max(final_probs, key=final_probs.get)
        return final_state, final_probs[final_state]



    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMM Generator and Viterbi Algorithm')
    parser.add_argument('--generate', type=int, metavar='N', help='Generate a sequence of N random observations')
    parser.add_argument('model', type=str, help='Path to the model basename (without .trans or .emit)')
    parser.add_argument('--forward', type=str, metavar='OBS_FILE', help='Compute the most likely final state for a given sequence of observations from OBS_FILE')

    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.model)

    if args.generate:
        observation = hmm.generate(args.generate)
        print(observation)

    if args.forward:
        with codecs.open(args.forward, 'r', 'utf-8') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
            for line in lines:
                obs = Observation([], line.split())
                final_state, prob = hmm.forward(obs)
                print(f"Most likely final state for the observation '{line}' is: {final_state} with probability {prob}")
