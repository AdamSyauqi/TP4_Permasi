import os
import pickle
import contextlib
import heapq
import time
import math
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')

from .index import InvertedIndexReader, InvertedIndexWriter
from .util import IdMap, sorted_merge_posts_and_tfs
from .compression import VBEPostings
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer


'''
Collaborator:
Andre Septiano - 1906398313
Matthew Tumbur Parluhutan - 1906308500
'''

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []
        self.stemmer = SnowballStemmer("english")
        self.stop_words = stopwords.words("english")


    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with InvertedIndexReader(
            directory = self.output_dir,
            index_name = self.index_name,
            postings_encoding = self.postings_encoding) as invert_map:
            self.dl_all = invert_map.doc_length
            temp = 0
            for doc in self.dl_all:
                temp += self.dl_all[doc]
            self.avdl = temp / len(self.dl_all)


    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        # TODO
        term_doc = set({})
        fileList = os.listdir(os.path.join(self.data_dir, block_dir_relative))

        for file in fileList:
            file_dir = os.path.join(block_dir_relative, file)
            text = open(os.path.join(self.data_dir, file_dir)).read()
            text = text.lower()
            words = nltk.word_tokenize(text)
            words = [self.stemmer.stem(token) for token in words]
            words = [token for token in words if token not in self.stop_words]

            doc_id = self.doc_id_map.__getitem__(file_dir)

            for word in words:
                term_id = self.term_id_map.__getitem__(word)
                term_doc.add((term_id, doc_id))
        return list(term_doc)

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}
        for term_id, doc_id in td_pairs:
            # print(term_id, doc_id)
            if term_id not in term_dict:
                term_dict[term_id] = {}
            try:
                term_dict[term_id][doc_id] += 1
            except KeyError:
                term_dict[term_id][doc_id] = 1

        for term_id in sorted(term_dict.keys()):
            # print(term_dict[term_id])
            sorted_tf = sorted(term_dict[term_id].items(), key=lambda kv: kv[0])
            # print(sorted_tf)
            postings_list = [k[0] for k in sorted_tf]
            tf_list = [k[1] for k in sorted_tf]
            index.append(term_id, postings_list, tf_list)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        self.load()

        terms = nltk.word_tokenize(query)
        terms = [self.stemmer.stem(token) for token in terms]
        terms = [term for term in terms if term not in self.stop_words]
        
        result = {}
        n = len(self.doc_id_map)

        with InvertedIndexReader(
            directory = self.output_dir,
            index_name = self.index_name,
            postings_encoding = self.postings_encoding) as invert_map:
            # print(invert_map.doc_length)
            for i in terms:
                if i not in self.term_id_map:
                    continue    
                
                postings_list, tf_list = invert_map.get_postings_list(self.term_id_map[i])
                for j in range(len(postings_list)):
                    doc = self.doc_id_map[postings_list[j]]
                    tf = tf_list[j]
                    df = invert_map.postings_dict[postings_list[j]][1]
                    wtq = math.log(n/df)
                    wtd = 0
                    if tf > 0:
                        wtd = 1 + math.log(tf)
                    if result.get(doc):
                        result[doc] = result[doc] + (wtd * wtq)
                    else:
                        result[doc] = (wtd * wtq)
        result = list(zip(result.values(), result.keys()))
        result.sort(key=lambda x: x[0], reverse=True)

        return result[:k]

    def retrieve_tfidf_binary_unary(self, query, k = 10):
        # TODO
        self.load()

        terms = nltk.word_tokenize(query)
        terms = [self.stemmer.stem(token) for token in terms]
        terms = [term for term in terms if term not in self.stop_words]
        
        result = {}
        n = len(self.doc_id_map)

        with InvertedIndexReader(
            directory = self.output_dir,
            index_name = self.index_name,
            postings_encoding = self.postings_encoding) as invert_map:
            # print(invert_map.doc_length)
            for i in terms:
                if i not in self.term_id_map:
                    continue    
                
                postings_list, tf_list = invert_map.get_postings_list(self.term_id_map[i])
                for j in range(len(postings_list)):
                    doc = self.doc_id_map[postings_list[j]]
                    tf = tf_list[j]
                    df = invert_map.postings_dict[postings_list[j]][1]
                    wtq = 1 # unary
                    wtd = 0
                    if tf > 0:
                        wtd = 1 # binary
                    if result.get(doc):
                        result[doc] = result[doc] + (wtd * wtq)
                    else:
                        result[doc] = (wtd * wtq)
        result = list(zip(result.values(), result.keys()))
        result.sort(key=lambda x: x[0], reverse=True)

        return result[:k]


    def bm_25(self, query, k = 10):
        # TODO
        self.load()

        terms = nltk.word_tokenize(query)
        terms = [self.stemmer.stem(token) for token in terms]
        terms = [term for term in terms if term not in self.stop_words]
        
        result = {}
        n = len(self.doc_id_map)

        k1 = 1
        k2 = 1.5

        b1 = 0.5
        b2 = 0.8

        '''
        wtd = (k1 + 1) * tf / k1 ((1-b1) + b1*dl/avdl) + tf
        '''

        with InvertedIndexReader(
            directory = self.output_dir,
            index_name = self.index_name,
            postings_encoding = self.postings_encoding) as invert_map:
            for i in terms:
                if i not in self.term_id_map:
                    continue    
                
                postings_list, tf_list = invert_map.get_postings_list(self.term_id_map[i])
                for j in range(len(postings_list)):
                    doc = self.doc_id_map[postings_list[j]]
                    try:
                        tf = tf_list[j]
                    except IndexError:
                        pass
                    
                    df = invert_map.postings_dict[postings_list[j]][1]
                    wtq = math.log(n/df)
                    wtd = 0
                    wtd1 = (k2 + 1) * tf
                    wtd2 = k2 * ((1 - b2) + b2 * self.dl_all[postings_list[j]] / self.avdl) + tf
                    wtd = wtd1 / wtd2
                    if result.get(doc):
                        result[doc] = result[doc] + (wtd * wtq)
                    else:
                        result[doc] = (wtd * wtq)
        result = list(zip(result.values(), result.keys()))
        result.sort(key=lambda x: x[0], reverse=True)

        return result[:k]


    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    try:
        os.mkdir("index")
    except:
        pass

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
