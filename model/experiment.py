import re
from bsbi import BSBIIndex
from compression import VBEPostings
import math
from letor import LetorClass
import numpy as np

######## >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan

      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
  """ menghitung search effectiveness metric score dengan
      Discounted Cumulative Gain

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan

      Returns
      -------
      Float
        score DCG
  """
  # TODO
  score = 0.
  for i in range(1, len(ranking) + 1):
      pos = i - 1
      score += ranking[pos] / math.log2(i + 1)
  return score

def ap(ranking):
  """ menghitung search effectiveness metric score dengan
      Average Precision

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan

      Returns
      -------
      Float
        score AP
  """
  # TODO
  score = 0.
  r = sum(ranking)
  for i in range(1, len(ranking) + 1):
      pos = i - 1
      score += ((sum(ranking[:i]) / len(ranking[:i])) / r) * ranking[pos]
  return score

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels)
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in BSBI_instance.bm_25(query, k = k):
        did = int(re.search(r'.*\/(.*)\.txt', doc).group(1))
        ranking.append(qrels[qid][did])
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))

  print("Hasil evaluasi BM25 terhadap 30 queries")
  print("RBP score = %.2f" % (sum(rbp_scores) / len(rbp_scores)))
  print("DCG score = %.2f" % (sum(dcg_scores) / len(dcg_scores)))
  print("AP score  = %.2f" % (sum(ap_scores) / len(ap_scores)))

def eval_letor(qrels, query_file = "queries.txt", k = 1000):

  letor = LetorClass()

  letor.do_train()

  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ap_scores = []
    for qline in file:
      X_unseen = []
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])
      print(query)

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      docs = []
      for (_, doc) in BSBI_instance.bm_25(query, k = k):
        text = open("collection/" + doc).read()
        text = text.lower()
        did = int(re.search(r'.*\/(.*)\.txt', doc).group(1))
        docs.append([did, text])
      
      print(docs[0])

      for _, doc in docs:
        X_unseen.append(letor.features(query.split(), doc.split()))

      X_unseen = np.array(X_unseen)
      score = letor.predict_ranker(X_unseen)
      did_scores = [x for x in zip([did for (did, _) in docs], score)]
      sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

      for (dids, score) in sorted_did_scores:
        ranking.append(qrels[qid][dids])

      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ap_scores.append(ap(ranking))


  print("Hasil evaluasi BM25 dengan Letor terhadap 30 queries")
  print("RBP score = %.2f" % (sum(rbp_scores) / len(rbp_scores)))
  print("DCG score = %.2f" % (sum(dcg_scores) / len(dcg_scores)))
  print("AP score  = %.2f" % (sum(ap_scores) / len(ap_scores)))

def eval_letor_content(k = 1000, queries = ["the crystalline lens in vertebrates, including humans.", \
           "compensatory renal hypertrophy - - stimulus resulting in mass increase (hypertrophy) and cell proliferation (hyperplasia) in the remaining kidney following unilateral nephrectomy in mammals.", \
           "methods for experimental production of and known causes of hydrocephalus in animals and humans."]):

  letor = LetorClass()

  letor.do_train()

  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

    # rbp_scores = []
    # dcg_scores = []
    # ap_scores = []
  for query in queries:
    print(query)
    X_unseen = []
      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      # ranking = []
    docs = []
    for (_, doc) in BSBI_instance.bm_25(query, k = k):
      text = open("collection/" + doc).read()
      text = text.lower()
      did = int(re.search(r'.*\/(.*)\.txt', doc).group(1))
      docs.append([did, text])

    print(docs[0])
    for _, doc in docs:
      X_unseen.append(letor.features(query.split(), doc.split()))

    X_unseen = np.array(X_unseen)
    score = letor.predict_ranker(X_unseen)
    did_scores = [x for x in zip([did for (did, _) in docs], score)]
    sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

    for (dids, score) in sorted_did_scores:
      print(dids, score)


if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  # Eval Letor
  #eval_letor(qrels)

  # Eval non-Letor
  #eval(qrels)

  eval_letor_content(10)
