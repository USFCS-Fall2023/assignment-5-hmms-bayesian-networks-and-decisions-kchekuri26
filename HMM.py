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
            next_state = np.random.choice(list(self.transitions[current_state].keys()),
                                          p=list(self.transitions[current_state].values()))
            states.append(next_state)

            emission = np.random.choice(list(self.emissions[next_state].keys()),
                                        p=list(self.emissions[next_state].values()))
            emissions.append(emission)

        return Observation(states[1:], emissions)

    def forward(self, observation):
        states = list(self.transitions.keys())
        observations = observation.outputseq

        T = len(observations)
        num_states = len(states)
        M = np.zeros((T, num_states))

        # initialization
        for s in range(num_states):
            state = states[s]
            if state in self.transitions['#'] and observations[0] in self.emissions[state]:
                M[0, s] = self.emissions[state][observations[0]] * self.transitions['#'][state]

        # Propagation
        for t in range(1, T):
            for s in range(num_states):
                sum_prob = 0
                for s2 in range(num_states):
                    prev_state = states[s2]
                    curr_state = states[s]
                    if curr_state in self.transitions[prev_state] and observations[t] in self.emissions[curr_state]:
                        sum_prob += M[t - 1, s2] * self.transitions[prev_state][curr_state] * \
                                    self.emissions[curr_state][observations[t]]
                M[t, s] = sum_prob

        final_state_index = np.argmax(M[-1])
        final_state = states[final_state_index]
        prob = M[-1, final_state_index]
        return final_state, prob

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
    parser.add_argument('--forward', type=str, metavar='OBS_FILE',
                        help='Compute the most likely final state for a given sequence of observations from OBS_FILE')

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
