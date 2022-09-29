import os
import ast
import re
import string
import contractions
from gensim.parsing.preprocessing import remove_stopwords
from ftfy import fix_encoding
from pywsd.utils import lemmatize_sentence

############################################################################### INITIALIZING VARIABLES
path = "enwiki20220701-stripped/AA/wiki_00"
rel_keys = ['id', 'text'] # keys that we find interesting

############################################################################## WHEN ITERATING OVER MULTIPLE FILES
# # for every file in the folder
# for f in os.listdir(path):
#     # open it
#     file = open(os.path.join(path + f), 'r')
#     # for every line convert it to a dictionary
##############################################################################

file = open(path, 'r')
for item in file:
    item = ast.literal_eval(item)
    
    # only keep texts that are not empty
    if item["text"]:
        # create new dictionary with only id and text as the keys
        filtered_data = {x:item[x] for x in rel_keys}
        # fix unicode
        filtered_data["text"] = fix_encoding(filtered_data["text"])
        # remove special characters (and also punctuation) except alphanumeric and parenthese
        filtered_data["text"] = re.sub("[^A-Za-z0-9']+", ' ',filtered_data["text"])
        # expand contractions
        filtered_data["text"] = contractions.fix(filtered_data['text'])
        # removing stopwords - does not remove stopwords with capitals I believe!!
        filtered_data["text"] = remove_stopwords(filtered_data["text"])
        # lemmatize
        filtered_data["text"] = lemmatize_sentence(filtered_data["text"])
        filtered_data["text"] = " ".join(filtered_data["text"])
        print(filtered_data)


        