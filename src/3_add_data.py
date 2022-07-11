import os
import time
import json
import numpy as np
import requests as req
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


API_KEY = os.environ['API_KEY']

headers = {'x-api-key': API_KEY}
params = {
    'fields': 'paperId,abstract,title,year,fieldsOfStudy,authors,tldr',
}

def make_base_url(paper_id:str) -> str:
    return f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}'

def make_request(paper_id:str):
    try:
        r = req.get(make_base_url(paper_id), headers=headers, params=params)
        time.sleep(0.1)
        resp = json.loads(r.content)
        if 'error' not in resp.keys():
            return {paper_id: resp}
    except Exception as e:
        print(f'Error occured: {e}')

def main():
    ids = np.load('../data/unique_ids_list.npy', allow_pickle=True)
    cit_rel = np.load('../data/citation_relations.npy', allow_pickle=True).item()

    executor = ThreadPoolExecutor(max_workers=os.cpu_count() + 4)
    with executor as pool:
        additional_data = list(tqdm(pool.map(make_request, ids), total=len(ids), desc='Scrapping additional data'))

    faulty_ids = []

    quick_lookup = {}
    for n, data in enumerate(additional_data):
        if 'error' in data.keys() or data['abstract'] is None:
            if ids[n] in cit_rel.keys():
                cit_rel.pop(ids[n])
            faulty_ids.append(n)
            continue

        try:
            quick_lookup[data['paperId']] = {
                'title': data['title'],
                'abstract': data['abstract'],
                'year': data['year'],
                'fieldsOfStudy': data['fieldsOfStudy'],
                'authors': data['authors'],
                'tldr': data['tldr']
            }
        except Exception as e:
            print(n, e)

    ids = np.delete(ids, faulty_ids)

    np.save('../data/unique_ids_list.npy', ids)
    np.save('../data/citation_relations.npy', cit_rel)
    np.save('../data/additional_info_dict.npy', quick_lookup)


if __name__ == '__main__':
    main()