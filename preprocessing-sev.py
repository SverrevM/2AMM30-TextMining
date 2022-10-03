import os
import ast
import re
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

############################################################################### INITIALIZING VARIABLES
path = "enwiki20220701-stripped/AA/"
rel_keys = ['id', 'text'] # keys that we find interesting
wn_lemmatizer = WordNetLemmatizer()
sw = stopwords.words('english')
""" 
REMARKS:
Create a folder called preprocessed with subfolders 'AA' and 'AB'
* words like game_77 now have become game77
* no punctuation left, so can't distinguish sentences anymore
* words that were put into parenthese and that included stopwords are treated like all the other words, 
meaning that it is also pre-processed and so information is probably lost, e.g. such as the title 'The last blabla that cannot' 
(the and cannot are gone, parenthese are gone)
* hyperlinks in text are not handled if present
"""

def preprocessing(path, f):
    file = open(path + f, 'r', encoding='utf-8')
    for item in file:
        item = ast.literal_eval(item)
        
        # only keep texts that are not empty
        if item["text"]:
            # create new dictionary with only id and text as the keys
            filtered_data = {x:item[x] for x in rel_keys}
            # remove special characters (and also punctuation) except alphanumeric, parenthese and whitespace
            filtered_data["text"] = re.sub("[^A-Za-z0-9' ]+", '',filtered_data["text"])
            # split data on whitespace
            filtered_data["text"] = filtered_data["text"].split(" ")
            # fix contractions if word in list is not empty
            filtered_data["text"] = [contractions.fix(w) for w in filtered_data["text"] if w]
            # join all words
            filtered_data["text"] = " ".join(filtered_data["text"])
            # get rid of the parenthese, replace by empty
            filtered_data["text"] = filtered_data["text"].replace("'", "")
            # split by whitespace again
            filtered_data["text"] = filtered_data["text"].split(" ")
            # filter out stopwords even the ones that are capitalized
            filtered_data["text"] = [w for w in filtered_data["text"] if w.lower() not in sw]
            # join words again
            filtered_data["text"] = " ".join(filtered_data["text"])
            # lemmatize
            filtered_data["text"] = [wn_lemmatizer.lemmatize(w) for w in filtered_data["text"].split(" ")]
            filtered_data["text"] = " ".join(filtered_data["text"])
            

            filtered_data = str(filtered_data)

            # write preprocessed file to a new file in the preprocessed folder (same structure, with the AA and AB folders)
            with open("preprocessed/AA/p_%s" % f, 'a') as preprocessed_file:
                preprocessed_file.write(filtered_data + '\n')
            preprocessed_file.close()


############################################################################## WHEN ITERATING OVER MULTIPLE FILES
# for every file in the folder
for f in os.listdir(path):
    preprocessing(path, f)
##############################################################################
        