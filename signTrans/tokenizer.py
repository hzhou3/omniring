# -*- coding: utf-8 -*-
"""
Sign Language Recognition Project

author: Hao Zhou

"""
import re, spacy

class Tokenizer:

    def __init__(self, lang: str = None):
        
        self.tokenizer = None
        
        self.lang = lang
        
        if lang == 'en':
            self.tokenizer = spacy.load('en_core_web_sm')
        elif lang == 'de':
            self.tokenizer = spacy.load('de_core_news_sm')
        else:
            self.tokenizer = None
            
        # print(self.lang)
        
            

    def split(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        
        text = text.lower()
        
        if self.lang == 'de' or self.lang == 'en':
            out = [tok.text for tok in self.tokenizer.tokenizer(text)]
            # print(out)
            return out
        
        else:
            text = re.sub('[^A-Za-z0-9 ]+', '', text)
            return text.split()
    
        
        
