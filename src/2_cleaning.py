import numpy as np


BERT_PAPER_ID = 'df2b0e26d0599ce3e70df8a9da02e51594e0e992'

def remove_excess_citations(data_indirect, bert_cit):
    cits = []

    for sample in data_indirect:
        if sample is not None:
            key = list(sample.keys())[0]
            d = {key: []}
            for cit in sample[key]:
                d[key].append(cit['citingPaper']['paperId'])
            cits.append(d)

    for cit in cits:
        key = list(cit.keys())[0]
        bert_second_level_cits = cit[key]
        for s_l_cit in bert_second_level_cits:
            if s_l_cit in bert_cit[BERT_PAPER_ID]:
                bert_cit[BERT_PAPER_ID].pop(bert_cit[BERT_PAPER_ID].index(s_l_cit))
    return cits

def calculate_unique_ids(cits):
    unique_ids = []
    for cit in cits:
        key = list(cit.keys())[0]
        unique_ids.append(key)
        unique_ids.extend(cit[key])
    return list(set(unique_ids))

def list_to_dict(l):
    d = {}
    for entry in l:
        key = list(entry.keys())[0]
        d[key] = entry[key]
    return d


def main():
    data_direct = np.load('../data/direct_citations.npy', allow_pickle=True)
    direct_cit_ids = [cit['citingPaper']['paperId'] for cit in data_direct.item()[BERT_PAPER_ID]]

    direct_cit_ids = list(set(direct_cit_ids))
    data_indirect = np.load('../data/indirect_citations.npy', allow_pickle=True)
    bert_cit = {BERT_PAPER_ID : direct_cit_ids}

    cits = remove_excess_citations(data_indirect, bert_cit)
    cits.append(bert_cit)

    unique_ids = calculate_unique_ids(cits)

    cits = list_to_dict(cits)

    np.save('../data/unique_ids_list.npy', np.array(list(set(unique_ids))))
    np.save('../data/citation_relations.npy', cits)


if __name__ == '__main__':
    main()