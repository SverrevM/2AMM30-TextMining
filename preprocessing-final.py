############################################################################### IMPORTS
import os
import spacy
import ftfy
import string
import json
import re
from os.path import isfile
############################################################################### INITIALIZING VARIABLES
path = "enwiki20220701-stripped/"
ner = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
nlp = spacy.load('en_core_web_sm') # for lemmatization and tokenization
############################################################################### RUN ONLY ONCE ELSE ERRORS
# merge entities when tokenizing text
nlp.add_pipe("merge_entities") 
contraction_map={
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "might have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "shall'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "will't've": "will not have",
    "would've": "would have",
    "would't": "would not",
    "y'all": "you all",
    "y'all'd": "you all would",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}
############################################################################### METHODS
def expand_contractions(sent, mapping):
    #pattern for matching contraction with their expansions
    pattern = re.compile('({})'.format('|'.join(mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    
    def expand_map(contraction):
        #using group method to access subgroups of the match
        match = contraction.group(0)
        #to retain correct case of the word
        first_char = match[0]
        #find out the expansion
        expansion = mapping.get(match) if mapping.get(match) else mapping.get(match.lower())
        expansion = first_char + expansion[1:]
        return expansion
    
    #using sub method to replace all contractions with their expansions for a sentence
    #function expand_map will be called for every non overlapping occurence of the pattern
    expand_sent = pattern.sub(expand_map, sent)
    return expand_sent

def preprocessing(path, pipeline):
    file = open(path, 'r').readlines()
    final_dictionary = dict()

    for item in file:
        fields = json.loads(item)
        # get entity labels based on NER
        title_entity_labels = [ent.label_ for ent in ner(fields["title"]).ents]

        # only keep texts that are not empty and filter on people (in title)
        if (fields["text"] and 'PERSON' in title_entity_labels):
            # impute encodings
            fields["text"] = ftfy.fix_text(fields["text"])
            # get rid of \xa0 - hard space or no-break space
            fields["text"] = fields["text"].replace(u'\xa0', u' ')
            # expand contractions
            fields["text"] = expand_contractions(fields["text"], contraction_map)
            # remove punctuation
            fields["text"] = fields["text"].translate(str.maketrans('', '', string.punctuation))

            if pipeline == "nltk":
            # OPT: lemmatize (for NLTK pipeline) 
                fields["text"] = [w.lemma_ for w in nlp(fields["text"])]
                fields["text"] = " ".join(fields["text"])
            
            # split article based on paragraphs into multiple entries in a new dictionary with their id: 1-1 (id=1, par=1) and so on. 
            custom_id = 1
            text_split = fields["text"].split('\n')
            for par in text_split:
                new_id = fields["id"] + "-" + str(custom_id)
                final_dictionary[new_id] = par
                custom_id = int(custom_id) + 1 

    return final_dictionary   

"""
        Function to apply preprocessing on a selection of files and store it in a separate folder
        - path: root folder (enwiki20220701-stripped), 
        - subf desired subfolder: AA or AB, must be passed as a string (e.g. "AA") 
        - files in subfolder AA: wiki_00 up till wiki_99
        - files in subfolder AB: wiki_00 up till wiki_68
        - start: start number file, e.g. 0-99 (no need to fill in 00, 0 is fine)
        - end: end number file, e.g. 0-99
        - the range is inclusive which means, e.g. with (0, 0) you select & pre-process file wiki_00,
        - with (32, 50) you select file wiki_32 up till wiki_50
        - pipeline: pass "nltk" or "rebel" to indicate for which pipeline you would like to preprocess, diff. 
          in preprocessing is the lemmatization. 
""" 
def preprocess_multiple_files(path, subf, start, end, pipeline): # (str, str, int, int, str)
    # if the folder preprocessed with its subfolders AA and AB resp. do not exist, create them
    if not os.path.exists("preprocessed-{pipeline}/AA".format(pipeline=pipeline)):
        os.makedirs("preprocessed-{pipeline}/AA".format(pipeline=pipeline))
    if not os.path.exists("preprocessed-{pipeline}/AB".format(pipeline=pipeline)):
        os.makedirs("preprocessed-{pipeline}/AB".format(pipeline=pipeline))

    # from file start to end
    for i in range(start, end+1):
        # to match the filename wiki_00 up till wiki_09 we add a zero in front of the number from user input if necessary
        if i < 10:
            i = "0" + str(i)

        # construct path to file name that falls within range
        f = path + subf + "/wiki_{}".format(i) 
        # if file exists
        if isfile(f):
            # process it
            p = preprocessing(f, pipeline)
            #TODO: FIX WHY SPECIAL ENCODINGS ARE NOT FIXED WHEN STORING
            # create a new file in the preprocessed folder, and put it into the concerning subfolder (AA or AB)
            with open("preprocessed-{pipeline}/{subf}/p_{p}_wiki_{nr}".format(subf=subf, nr=i, pipeline=pipeline, p=pipeline[0]), 'w') as preprocessed_file:
                preprocessed_file.write(json.dumps(p))
            preprocessed_file.close()

############################################################################## EXECUTION

preprocess_multiple_files(path, "AA", 0, 1, "rebel")


