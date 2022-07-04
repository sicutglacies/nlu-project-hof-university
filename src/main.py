import numpy as np
from tqdm import tqdm
from neomodel import StructuredNode, StringProperty, RelationshipTo, RelationshipFrom, config


config.DATABASE_URL = 'bolt://neo4j:neo4j123@192.168.91.18:33059'

N_PUB_TO_DISPLAY = 24811 # 24811

class Publication(StructuredNode):
    paper_id = StringProperty(unique_index=True)
    title = StringProperty()
    abstract = StringProperty()
    summary = StringProperty()
    authors = StringProperty()
    year_of_publication = StringProperty()
    field_of_study = RelationshipTo("FieldOfStudy", "INVOLVES")
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

class ClassContainer:
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


data_files = ['act_functions', 'archs', 'data_types', 'languages', 'topics']
classes = [ActFun, Arch, DataType, Language, FieldOfStudy]
cc = ClassContainer(data_files)

for n, (file, clazz) in enumerate(zip(data_files, classes)):
    entries = np.load(f'../data/lists/{file}.npy', allow_pickle=True)
    for entry in entries:
        class_sample = clazz(name=entry).save()
        cc.add_sample(file, entry, class_sample)

publications_dict = {}

publication_ids = np.load('../data/lists/unique_ids_list.npy', allow_pickle=True)
publication_data = np.load('../data/lists/full_data_dict.npy', allow_pickle=True)

abandoned_pubs = 0

for pub_id in tqdm(publication_ids[:N_PUB_TO_DISPLAY]):
    counter = 0
    d = publication_data.item()[pub_id]

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
    ).save()

    publications_dict[pub_id] = pub

    if d['language'] is not None:
        for lang in d['language']:
            pub.language.connect(cc.get_item('languages', lang))
            counter += 1
    
    if d['fieldsOfStudy'] is not None:
        for field in d['fieldsOfStudy']:
            pub.field_of_study.connect(cc.get_item('topics', field))
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

cit_lookup = np.load('../data/lists/cit_rel_dict.npy', allow_pickle=True).item()
cited_keys = list(cit_lookup.keys())
keys = list(publications_dict.keys())

n_connections = 0
for key in tqdm(keys):
    if key in cited_keys:
        cit_ids = cit_lookup[key]
        for iid in cit_ids:
            if iid in keys:
                publications_dict[key].citing.connect(publications_dict[iid])
                n_connections += 1

print(f'Total {n_connections} citing connections were made')
print(f'Total {abandoned_pubs} publications did not have any connections')
