import re
from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LetorClass
import numpy as np
import joblib

def eval_letor_content(k = 10, query = "the crystalline lens in vertebrates, including humans."):

    letor = LetorClass()

    letor.dictionary = joblib.load("dict.pkl")
    letor.ranker = joblib.load("ranker.pkl")
    letor.model = joblib.load("model.pkl")

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    X_unseen = []

    docs = []

    for (_, doc) in BSBI_instance.bm_25(query, k=k):
        text = open("collection/" + doc).read()
        text = text.lower()
        did = int(re.search(r'.*\/(.*)\.txt', doc).group(1))
        docs.append([did, text])

    for _, doc in docs:
        X_unseen.append(letor.features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)
    score = letor.predict_ranker(X_unseen)
    did_scores = [x for x in zip([did for (did, _) in docs], score)]
    sorted_did_scores = sorted(did_scores, key=lambda tup: tup[1], reverse=True)

    for (dids, score) in sorted_did_scores:
        print(dids, score)


if __name__ == '__main__':
    eval_letor_content()
