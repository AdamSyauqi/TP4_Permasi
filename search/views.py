from django.shortcuts import render
from .model import eval_letor_content

import sys
import os

# Create your views here.

def index(request):
    query = request.GET.get('search_bar')
    if query == None or query == "":
        context = {
            'query': query,
        }
        return render(request, 'search/index.html', context)
    else:
        result = {}
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        
        result_raw = eval_letor_content(100, query)
        for (doc_id, _) in result_raw:
            col = int(doc_id)//100 + 1
            text_file = os.path.dirname(__file__) + "/collection/" + str(col) + "/" + str(doc_id) +".txt"
            text = open(text_file).read()
            text = text.lower()
            result[doc_id] = text

        context = {
            'result': result,
            'query': query,
        }
        return render(request, 'search/index.html', context)

def content(request, doc_id):
    col = int(doc_id)//100 + 1
    text_file = os.path.dirname(__file__) + "/collection/" + str(col) + "/" + str(doc_id) +".txt"
    text = open(text_file).read()
    text = text.lower()

    context = {
        'doc_id': doc_id,
        'text': text,
    }
    return render(request, 'search/content.html', context)