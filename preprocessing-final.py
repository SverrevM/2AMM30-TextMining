import os
import spacy
import ftfy
import string
import json
import re
import pickle
############################################################################### INITIALIZING VARIABLES
path = "enwiki20220701-stripped/AA/"
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
###############################################################################
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

def preprocessing(path, f):
    file = open(path + f, 'r').readlines()
    final_dictionary = dict()

    for item in file:
        fields = json.loads(item)
        # get entity labels based on NER
        first_paragraph = fields["text"].split('\n')[0]
        paragraph_entity_labels = [ent.label_ for ent in ner(first_paragraph).ents]
        title_entity_labels = [ent.label_ for ent in ner(fields["title"]).ents]

        # only keep texts that are not empty and filter on people (in title or in first paragraph of the article)
        if (fields["text"] and 'PERSON' in title_entity_labels) or ('PERSON' in paragraph_entity_labels):
            # impute encodings
            fields["text"] = ftfy.fix_text(fields["text"])
            # expand contractions
            fields["text"] = expand_contractions(fields["text"], contraction_map)
            # remove punctuation
            fields["text"] = fields["text"].translate(str.maketrans('', '', string.punctuation))

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


def createFile(dict):
    with open("preprocessed/AA/p_%s" % f, 'w') as preprocessed_file:
        preprocessed_file.write(json.dumps(dict))
    preprocessed_file.close()
############################################################################## WHEN ITERATING OVER MULTIPLE FILES   
# for every file in the folder
# for f in os.listdir(path):
f = "wiki_00"
d = preprocessing(path, f)
createFile(d)
##############################################################################
