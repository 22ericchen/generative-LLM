from collections import Counter
import numpy as np
import math
import random

"""
CS 4120, Spring 2024
Homework 3 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
    Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

    Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
    """
    # STUDENTS IMPLEMENT 
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def read_file(path: str) -> list:
    """
    Reads the contents of a file in line by line.
    Args:
    path (str): the location of the file to read

    Returns:
    list: list of strings, the contents of the file
    """
    # PROVIDED
    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
    """
    Tokenize a single string. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
    False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

    Returns:
    list of strings - a single line tokenized
    """
    # PROVIDED
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
    # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
    """
    Tokenize each line in a list of strings. Glue on the appropriate number of 
    sentence begin tokens and sentence end tokens (ngram - 1), except
    for the case when ngram == 1, when there will be one sentence begin
    and one sentence end token.
    Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

    Returns:
    list of strings - all lines tokenized as one large list
    """
    # PROVIDED
    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram):
        """Initializes an untrained LanguageModel
        Args:
          n_gram (int): the n-gram order of the language model to create
        """
        # STUDENTS IMPLEMENT
        self.n_gram = n_gram
        self.ngram_counts = Counter()
        self.total_counts = 0 
        self.vocab = set()
        self.prefix_counts = Counter()
        
       
  
    def train(self, tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Args:
          tokens (list): tokenized data to be trained on as a single list
          verbose (bool): default value False, to be used to turn on/off debugging prints
        """
        # STUDENTS IMPLEMENT
        
        #determine which tokens only exist once and replace to UNK
        token_counts = Counter(tokens)      
        tokens = [token if token_counts[token] > 1 else UNK for token in tokens]
            
        #create and store ngrams and their respective counts
        all_ngrams = create_ngrams(tokens, self.n_gram)
        self.ngram_counts = Counter(all_ngrams)
        
        #number of all tokens in corpus
        self.total_counts = sum(self.ngram_counts.values())
        
        #store and count all contexts of each ngram
        if self.n_gram > 1:
            for ngram in all_ngrams:
                self.prefix_counts[ngram[:-1]] += 1
        
        #define the vocab of the corpus
        self.vocab = set(tokens)
        
        if verbose:
            print(f"Trained on {self.total_counts} total n-grams")
    

    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
        Args:
          sentence_tokens (list): a tokenized sequence to be scored by this model
      
        Returns:
          float: the probability value of the given tokens for this model
        """
        # STUDENTS IMPLEMENT
        
        #set initial probability
        probability = 1.0
        
        #change all unexpected tokens to UNK
        updated_tokens = [UNK if token not in self.vocab else token for token in sentence_tokens]
        
        #create the ngrams
        sentence_ngrams = create_ngrams(updated_tokens, self.n_gram)
        vocab_size = len(set(sentence_ngrams))
        
        for ngram in sentence_ngrams:
            ngram_count = self.ngram_counts[ngram]            
            prefix_count = 0
            
            if self.n_gram == 1:
                prefix_count = self.total_counts
            else:
                prefix_ngram = ngram[:-1]
                prefix_count = self.prefix_counts[prefix_ngram]                
            probability *= ((ngram_count + 1) / (prefix_count + len(self.vocab)))
            
        return probability
        
    
    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.
      
        Returns:
          list: the generated sentence as a list of tokens
        """
        # STUDENTS IMPLEMENT        
        #initialize setence structure
        sentence = [SENTENCE_BEGIN]
        
        #determine if model is a unigram or not
        if self.n_gram > 1:
            #construct first n-1 context gram
            while len(sentence) < self.n_gram - 1:
                possible_next_tokens = [ngram[1] for ngram in self.ngram_counts.keys() if ngram[:len(sentence)] == tuple(sentence)]
                possible_next_tokens = [token for token in possible_next_tokens if token !='<s>' and token != UNK]
                #if no possible tokens end the loop and continue
                if not possible_next_tokens:
                    break  # Break if no possible continuations
                #choose one of the possible tokens to build the context    
                next_token = random.choice(possible_next_tokens)
                sentence.append(next_token)
            
            #loop to generate ngram sentence
            while sentence[-1] != SENTENCE_END:                
                #determine the context
                prefix = tuple(sentence[-(self.n_gram - 1):])
                #identify possible grams
                possible_grams = [ngram for ngram in self.ngram_counts if ngram[:-1] == prefix]
                possible_grams = [ngram for ngram in possible_grams if ngram[-1] != UNK]

                #if no possible continuations, end the sentence
                if not possible_grams:
                    sentence.append(SENTENCE_END)
                    break

                #calculate a weighted list of possible next words
                weights = [self.ngram_counts[gram] for gram in possible_grams]
                #randomly select a choice based on weighted possibility
                next_gram = random.choices(possible_grams, weights=weights, k=1)[0]
                next_token = next_gram[-1]
                
                #add to the sentence
                sentence.append(next_token)
        else:
            #loop to construct unigram model sentence
            while sentence[-1] != SENTENCE_END:
                possible_gram = [gram for gram in self.ngram_counts.keys() if gram != ('<s>',)]
                weights = [self.ngram_counts[gram] for gram in possible_gram]
                next_word = random.choices(possible_gram, weights=weights, k=1)[0]
                sentence.extend(next_word)
                
        return sentence

    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
        Args:
          n (int): the number of sentences to generate
      
        Returns:
          list: a list containing lists of strings, one per generated sentence
        """
        # PROVIDED
        return [self.generate_sentence() for i in range(n)]


    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
        Args:
          sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
        Returns:
          float: the perplexity value of the given sequence for this model
        """
        # STUDENTS IMPLEMENT
        
        #number of words
        N = len(sequence)
        #get the probability of the sequence given the trained model
        probability = self.score(sequence)
        if probability > 0:
            perplexity = math.pow(probability, -1/N)
        else:
            perplexity = float('inf')

        return perplexity
  
    # not required
if __name__ == '__main__':
    print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")

