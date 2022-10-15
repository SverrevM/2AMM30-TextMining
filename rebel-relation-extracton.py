############################################################################### IMPORTS
import requests
import re
import hashlib
from spacy import Language
from typing import List
from spacy.tokens import Doc, Span
from transformers import pipeline
import crosslingual_coreference
import spacy
from os.path import isfile
import os
import ftfy
import json
import glob
############################################################################### INITIALIZE VARIABLES
path = "preprocessed-rebel/"
############################################################################### METHODS
def call_wiki_api(item):
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        return data['search'][0]['id']
    except:
        return 'id-less'

def extract_triplets(text):
    """
    Function to parse the generated text and extract the triplets
    """
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})

    return triplets

@Language.factory(
    "rebel",
    requires=["doc.sents"],
    assigns=["doc._.rel"],
    default_config={
        "model_name": "Babelscape/rebel-large",
        "device": 0,
    },
)
class RebelComponent:
    def __init__(
        self,
        nlp,
        name,
        model_name: str,
        device: int,
    ):
        assert model_name is not None, ""
        self.triplet_extractor = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
        self.entity_mapping = {}
        # Register custom extension on the Doc
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default={})
   
    def get_wiki_id(self, item: str):
        mapping = self.entity_mapping.get(item)
        if mapping:
            return mapping
        else:
            res = call_wiki_api(item)
            self.entity_mapping[item] = res
            return res

    def _generate_triplets(self, sent: Span) -> List[dict]:
        output_ids = self.triplet_extractor(sent.text, return_tensors=True, return_text=False)[0]["generated_token_ids"]["output_ids"]
        extracted_text = self.triplet_extractor.tokenizer.batch_decode(output_ids[0])
        extracted_triplets = extract_triplets(extracted_text[0])
        return extracted_triplets

    def set_annotations(self, doc: Doc, triplets: List[dict]):
        for triplet in triplets:

            # Remove self-loops (relationships that start and end at the entity)
            if triplet['head'] == triplet['tail']:
                continue

            # Use regex to search for entities
            head_span = re.search(triplet["head"], doc.text)
            tail_span = re.search(triplet["tail"], doc.text)

            # Skip the relation if both head and tail entities are not present in the text
            # Sometimes the Rebel model hallucinates some entities
            if not head_span or not tail_span:
                continue

            index = hashlib.sha1("".join([triplet['head'], triplet['tail'], triplet['type']]).encode('utf-8')).hexdigest()
            if index not in doc._.rel:
                # Get wiki ids and store results
                doc._.rel[index] = {"relation": triplet["type"], "head_span": {'text': triplet['head'], 'id': self.get_wiki_id(triplet['head'])}, "tail_span": {'text': triplet['tail'], 'id': self.get_wiki_id(triplet['tail'])}}

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            sentence_triplets = self._generate_triplets(sent)
            self.set_annotations(doc, sentence_triplets)
        return doc

DEVICE = -1 # Number of the GPU, -1 if want to use CPU
# Add coreference resolution model
coref = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
coref.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": DEVICE})

# Define rel extraction model
rel_ext = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer', 'attribute_rules', 'tagger'])
rel_ext.add_pipe("rebel", config={
    'device':DEVICE, # Number of the GPU, -1 if want to use CPU
    'model_name':'Babelscape/rebel-large'} # Model used, will default to 'Babelscape/rebel-large' if not given
    )


def coreff(par):
    # process it
    coref_text = coref(par)._.resolved_text
    doc = rel_ext(coref_text)  
    for value, rel_dict in doc._.rel.items():
        print(f"{value}: {rel_dict}")

"""
        Function to apply preprocessing on a selection of files and store it in a separate folder
        - path: root folder (preprocessed-rebel), 
        - subf desired subfolder: AA or AB, must be passed as a string (e.g. "AA") 
        - files in subfolder AA: e.g. p_r_wiki_00 
        - files in subfolder AB: e.g. p_r_wiki_00
        - start: start number file, e.g. 0-99 (no need to fill in 00, 0 is fine)
        - end: end number file, e.g. 0-99
        - the range is inclusive which means, e.g. with (0, 0) you select & pre-process file wiki_00,
        - with (32, 50) you select file wiki_32 up till wiki_50
""" 
def rel_extraction_mul_files(path, subf=None, start=None, end=None):
    if subf:
        # from file start to end
        for i in range(start, end+1):
            # to match the filename p_r_wiki_00 up till p_r_wiki_09 we add a zero in front of the number from user input if necessary
            if i < 10:
                i = "0" + str(i)
            # construct path to file name that falls within range
            f = path + subf + "/p_r_wiki_{}".format(i) 
            
            # if file exists
            if isfile(f):
                # OPEN FILE, GO THROUGH EACH SEN AND PASS THAT INTO COREF FUNCTION
                file = open(f, 'r', encoding='utf-8')
                doc = json.load(file)
                for k, v in doc.items():
                    v = ftfy.fix_text(v) # FIX ANY ENCODINGS
                    # if paragraph has more than one word
                    if len(v.split(" ")) > 1:
                        coreff(doc[k])
                        #TODO: FORMAT OUTPUT -> KADIR
    else:
        for f in glob.glob('preprocessed-rebel/*/*'):
            # OPEN EACH FILE, GO THROUGH EACH PARAGRAPH AND PASS THAT INTO COREF FUNC
            file = open(f, 'r', encoding='utf-8')
            doc = json.load(file)
            for k, v in doc.items():
                v = ftfy.fix_text(v) # FIX ANY ENCODINGS
                # if paragraph has more than one word
                if len(v.split(" ")) > 1:
                    coreff(doc[k])
                    #TODO: FORMAT OUTPUT -> KADIR
############################################################################ EXECUTION
# OPT 1: SELECT SPECIFIC FILES TO FEED REBEL IN ONE PARTICULAR MAP / # OPT 2: FEED ALL FILES, in AA and AB BY ONLY KEEPING PATH IN THERE
rel_extraction_mul_files(path, "AA", 0, 0) # rel_extraction_mul_files(path)
