#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 23:33:28 2022

@author: alexloubser
"""
#data
import unicodedata
import string
class TextProcess:
    def __init__(self):
        char_map_str = """
		<PAD> 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
        ' 28
        <UNK> 29
		"""
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def unicode_to_ascii(self,s):
        ALL_LETTERS = string.ascii_letters + " '"
        return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)
        
    def text_to_int_sequence(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ' or c== "-": 
                ch = self.char_map['<SPACE>']  
            else:
                ch = self.char_map[c.lower()] 
            int_sequence.append(ch)
        return int_sequence
   
    def int_to_text_sequence(self, labels):
        """Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')
    