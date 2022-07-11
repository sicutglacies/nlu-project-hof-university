import os
import time
import json
import numpy as np
import requests as req
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


API_KEY = os.environ['API_KEY']
BERT_PAPER_ID = 'df2b0e26d0599ce3e70df8a9da02e51594e0e992'

headers = {'x-api-key': API_KEY}
params = {
    'fields': 'abstract,url,title,citationCount',
}

def make_cit_url(paper_id:str) -> str:
    return f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations' 

def make_base_url(paper_id:str) -> str:
    return f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}'

def download_direct_citations(paper_id:str):
    try:
        params = {
            'fields': 'abstract,url,title,citationCount',
        }
        params['offset'] = '0'
        r = req.get(make_base_url(paper_id), headers=headers, params=params)
        cit_count = json.loads(r.content)['citationCount']
        max_cit_count = cit_count
        chunk_size = 0
        if cit_count > 9999: max_cit_count = 9999
        if max_cit_count > 1000:
            chunk_size = 1000
        else:
            chunk_size = max_cit_count

        data = []
        params['limit'] = str(chunk_size)
        while len(data) < max_cit_count:
            diff = max_cit_count - len(data)
            if diff < int(params['limit']):
                params['limit'] = str(diff)
            time.sleep(0.1)
            r = req.get(make_cit_url(paper_id), headers=headers, params=params)
            j = json.loads(r.content)
            data.extend(j['data'])
            if 'next' in j.keys(): 
                params['offset'] = str(j['next'])
        return {paper_id: data}
    except Exception as e:
        print(f'Problem occured with {e}')

def main():
    print('Scrapping direct citations..')
    data = download_direct_citations(BERT_PAPER_ID)
    np.save('../data/direct_citations.npy', data)

    citations_paper_ids = list(set([cit['citingPaper']['paperId'] for cit in data[BERT_PAPER_ID]]))

    executor = ThreadPoolExecutor(max_workers=os.cpu_count() + 4)
    with executor as pool:
        data_indirect = list(tqdm(pool.map(download_direct_citations, citations_paper_ids), total=len(citations_paper_ids), desc='Scrapping indirect citations'))

    np.save('../data/indirect_citations.npy', data_indirect)


if __name__ == '__main__':
    main()
