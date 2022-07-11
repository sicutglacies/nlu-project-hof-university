import os
import numpy as np
from tqdm import tqdm # The tqdm module is designed for quick and scalable implementation of progress indicators, providing end users with a visual indication of the progress of a calculation or data transfer
from neomodel import StructuredNode, StringProperty, RelationshipTo, RelationshipFrom, config
# StructuredNode - The basic package for working with neo4j. A compulsory class to inherit from the model

LOGIN = os.environ['LOGIN']
PASS = os.environ['PASS']

config.DATABASE_URL = f'bolt://{LOGIN}:{PASS}@192.168.91.18:33059' # This is database link

publications_dict = {}
publication_ids = np.load('../data/unique_ids_list.npy', allow_pickle=True) # loading id
publication_data = np.load('../data/full_data_dict.npy', allow_pickle=True) # all data for publication

N_PUB_TO_DISPLAY = len(publication_ids) # limit the number if needed

class Publication(StructuredNode):
    paper_id = StringProperty(unique_index=True) # Fields
    title = StringProperty()
    abstract = StringProperty()
    summary = StringProperty()
    authors = StringProperty()
    year_of_publication = StringProperty()
    field_of_study = RelationshipTo("FieldOfStudy", "INVOLVES") # realationship(to FieldOfStudy(Node), the realationship is called INVOLVES)
    citing = RelationshipFrom("Publication", "CITING")
    works_with_datatype = RelationshipTo("DataType", "WORKS-WITH-DATA")
    language = RelationshipTo('Language', 'TRAINED-ON')
    architecture = RelationshipTo('Arch', 'USES-TECHNOLOGY')
    act_fun = RelationshipTo('ActFun', 'USES-ACT-FUNCTION')

class FieldOfStudy(StructuredNode):
    name = StringProperty(unique_index=True)

class DataType(StructuredNode):
    name = StringProperty(unique_index=True)

class Language(StructuredNode):
    name = StringProperty(unique_index=True)

class Arch(StructuredNode):
    name = StringProperty(unique_index=True)

class ActFun(StructuredNode):
    name = StringProperty(unique_index=True)

class ClassContainer: # data recorder class
    def __init__(self, names):
        self.names = names
        self.storage = {}
        for name in names:
            self.storage[name] = {}
    
    def add_sample(self, name, subname, class_sample):
        if name == 'act_functions':
            subname = subname.lower()
        self.storage[name][subname] = class_sample

    def get_item(self, name, subname):
        return self.storage[name][subname]


data_files = ['act_functions', 'archs', 'data_types', 'languages', 'topics'] # name of files
classes = [ActFun, Arch, DataType, Language, FieldOfStudy] # record classes
cc = ClassContainer(data_files) # container is created with colum(node) names

for n, (file, clazz) in enumerate(zip(data_files, classes)): # sort of files and classes 
    entries = np.load(f'../data/{file}.npy', allow_pickle=True) # downloading data from files
    for entry in entries:
        class_sample = clazz(name=entry).save()
        cc.add_sample(file, entry, class_sample) # adding data to the container

abandoned_pubs = 0

for pub_id in tqdm(publication_ids[:N_PUB_TO_DISPLAY]): # sort publication_ids
    counter = 0
    d = publication_data.item()[pub_id] # retrieved from publication_data

    try:
        summary = str(d['tldr']['text'])
    except Exception as e:
        summary = 'No summary'

    pub = Publication(
        paper_id = pub_id,
        title = str(d['title']),
        abstract = str(d['abstract']),
        authors = str(d['authors_string']),
        year_of_publication = str(d['year']),
        summary = summary
    ).save() # Take the data from what we received and write it into the Publication entity

    publications_dict[pub_id] = pub

    if d['language'] is not None: # if we have a non-empty value for language
        for lang in d['language']:
            pub.language.connect(cc.get_item('languages', lang)) # Creating a relationship to language 
            counter += 1
    
    if d['fieldsOfStudy'] is not None:
        for field in d['fieldsOfStudy']:
            pub.field_of_study.connect(cc.get_item('topics', field)) # Creating a relationship to fieldsOfStudy
            counter += 1

    if d['architecture'] is not None:
        for arch in d['architecture']:
            pub.architecture.connect(cc.get_item('archs', arch))
            counter += 1

    if d['data_type'] is not None:
        for dt in d['data_type']:
            pub.works_with_datatype.connect(cc.get_item('data_types', dt))

    if d['act_function'] is not None:
        for act_f in d['act_function']:
            act_f = act_f.lower()
            pub.act_fun.connect(cc.get_item('act_functions', act_f))
            counter += 1

    if counter == 0:
        abandoned_pubs += 1

cit_lookup = np.load('../data/cit_rel_dict.npy', allow_pickle=True).item() # uploading the file for quotes(cit)
cited_keys = list(cit_lookup.keys())
keys = list(publications_dict.keys())

n_connections = 0
for key in tqdm(keys):
    if key in cited_keys:
        cit_ids = cit_lookup[key]
        for iid in cit_ids:
            if iid in keys:
                publications_dict[key].citing.connect(publications_dict[iid]) # keeping the relationship to the quotes(cit)
                n_connections += 1

print(f'Total {n_connections} citing connections were made')
print(f'Total {abandoned_pubs} publications did not have any connections')
