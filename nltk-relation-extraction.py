######################################################################################## IMPORTS
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.sem.relextract import extract_rels
from os.path import isfile
import json
import ftfy
import glob
import re
############################################################################### INITIALIZING VARIABLES
path = "preprocessed-nltk/"
######################################################################################## RUN ONLY ONCE
# download the pos tagger since its not included by default
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# enable only if fresh NLTK or lots of issues
nltk.download('all')
######################################################################################## METHODS
"""
        Function to apply preprocessing on a selection of files and store it in a separate folder
        - path: root folder (preprocessed-nltk), 
        - subf desired subfolder: AA or AB, must be passed as a string (e.g. "AA") 
        - files in subfolder AA: e.g. p_n_wiki_00 
        - files in subfolder AB: e.g. p_n_wiki_00
        - start: start number file, e.g. 0-99 (no need to fill in 00, 0 is fine)
        - end: end number file, e.g. 0-99
        - the range is inclusive which means, e.g. with (0, 0) you select & pre-process file wiki_00,
        - with (32, 50) you select file wiki_32 up till wiki_50
""" 
def rel_extraction_mul_files(path, subf=None, start=None, end=None):
    if subf:
        # from file start to end
        for i in range(start, end+1):
            # to match the filename p_n_wiki_00 up till p_n_wiki_09 we add a zero in front of the number from user input if necessary
            if i < 10:
                i = "0" + str(i)
            # construct path to file name that falls within range
            f = path + subf + "/p_n_wiki_{}".format(i) 
            
            # if file exists
            if isfile(f):
                # OPEN FILE, GO THROUGH EACH SEN AND PASS THAT INTO COREF FUNCTION
                file = open(f, 'r', encoding='utf-8')
                doc = json.load(file)
                for k, v in doc.items():
                    v = ftfy.fix_text(v) # FIX ANY ENCODINGS
                    # if paragraph has more than one word
                    if len(v.split(" ")) > 1: # TODO: CODE NOT REALLY GIVING DESIRED OUTPUT? 
                        te = v.splitlines()
                        use = te[0]
                        # tokenize sentence
                        token = word_tokenize(use)
                        # pos_tag sentence
                        pos = pos_tag(token)
                        # chunk sentence
                        chunked = ne_chunk(pos)
                        # (any of 'LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION', 'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE')
                        rels = extract_rels('PERSON', 'GPE', chunked, corpus='ace', pattern=re.compile(r'.*\bborn\b.*'), window = 100)
                        print(rels) 
                        #TODO: FORMAT OUTPUT -> KADIR
    else:
        for f in glob.glob('preprocessed-nltk/*/*'):
            # OPEN EACH FILE, GO THROUGH EACH PARAGRAPH AND PASS THAT INTO COREF FUNC
            file = open(f, 'r', encoding='utf-8')
            doc = json.load(file)
            for k, v in doc.items():
                v = ftfy.fix_text(v) # FIX ANY ENCODINGS
                # if paragraph has more than one word
                if len(v.split(" ")) > 1:
                    pass #TODO: ADD SAME CODE HERE
                    #TODO: FORMAT OUTPUT -> KADIR
############################################################################ EXECUTION
# OPT 1: SELECT SPECIFIC FILES TO FEED NLTK IN ONE PARTICULAR MAP / # OPT 2: FEED ALL FILES, in AA and AB BY ONLY KEEPING PATH IN THERE
rel_extraction_mul_files(path, "AA", 0, 0) # rel_extraction_mul_files(path)
