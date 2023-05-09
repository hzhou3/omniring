# -*- coding: utf-8 -*-
"""
Sign Language Recognition Project

author: Hao Zhou

"""


import json, os
from collections import defaultdict
from typing import List
import numpy as np


from signTrans.Constants import (                       
                            UNK_TOKEN,
                            PAD_TOKEN,
                            SOS_TOKEN,
                            EOS_TOKEN,
                            )


class VocabularyBase:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self):
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def _from_list(self, tokens: List[str] = None):
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.
        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str):
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.
        :param file: path to file where the vocabulary is loaded from
        """

        file = os.path.abspath(__file__)
        file = file.replace('signTrans', 'data')
        file = file.replace('utils', '')
        file = file.replace('vocabulary.py', 'vocab.json')
    
        if os.path.exists(file):
            print("Loading from vocab.json file")
            
                
            with open(file) as json_file:
                vocab_dict = json.load(json_file)
            json_file.close()
            
            tokens = []
            
            for k, v in vocab_dict.items():
                tokens.append(k)
            self._from_list(tokens)
        else:
            print("vocab file not exist!")

            

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str):
        """
        Save the vocabulary to a file, by writing token with index i in line i.
        :param file: path to file where the vocabulary is written
        """
        with open(file, "w", encoding="utf-8") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]):
        """
        Add list of tokens to vocabulary
        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            if t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary
        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)









class Vocabulary(VocabularyBase):
    def __init__(self,
                 tokens: List[str] = None,
                 file: str = None,
                 min_fre: int = 0):
        
        # TODO: filter out based on min_fre
        """
        Create vocabulary from list of tokens or file.
        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.
        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        super().__init__()
        self.specials = [UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, SOS_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 0
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)
        self.min_fre = min_fre
        
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.
        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.
        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences




    
if __name__ == '__main__':
    pass
    