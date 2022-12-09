import re
from .bsbi import BSBIIndex
from .compression import VBEPostings
from .letor import LetorClass
import numpy as np
import joblib
import sys
import os


def eval_letor_content(k = 100, query = "the crystalline lens in vertebrates, including humans."):

    letor = LetorClass()
    this_dir = os.path.dirname(__file__)

    letor.dictionary = joblib.load(os.path.join(this_dir, "dict.pkl"))
    letor.ranker = joblib.load(os.path.join(this_dir, "ranker.pkl"))
    letor.model = joblib.load(os.path.join(this_dir, "model.pkl"))

    BSBI_instance = BSBIIndex(data_dir=os.path.join(this_dir, 'collection'),
                              postings_encoding=VBEPostings,
                              output_dir=os.path.join(this_dir,'index'))

    X_unseen = []

    docs = []

    for (_, doc) in BSBI_instance.bm_25(query, k=k):
        text = open(os.path.join(this_dir, "collection/") + doc).read()
        text = text.lower()
        did = int(re.search(r'.*\/(.*)\.txt', doc).group(1))
        docs.append([did, text])

    if len(docs) < 1:
        return None

    else:
        for _, doc in docs:
            X_unseen.append(letor.features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)
        score = letor.predict_ranker(X_unseen)
        did_scores = [x for x in zip([did for (did, _) in docs], score)]
        sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

        return sorted_did_scores

if __name__ == '__main__':
    eval_letor_content()
