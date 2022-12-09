from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages


from .model import eval_letor_content

import sys
import os


# Create your views here.

def index(request):
    query = request.GET.get('search_bar')
    if query == None or query == "":
        context = {
            'result': result,
            'query': query,
        }
        return render(request, 'search/index.html', context)
    else:
        result = {}
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        
        result_raw = eval_letor_content(100, query)
        for (doc_id, _) in result_raw:
            col = None
            if doc_id < 100:
                col = 1
            elif doc_id < 200:
                col = 2
            elif doc_id < 300:
                col = 3
            elif doc_id < 400:
                col = 4
            elif doc_id < 500:
                col = 5
            elif doc_id < 600:
                col = 6
            elif doc_id < 700:
                col = 7
            elif doc_id < 800:
                col = 8
            elif doc_id < 900:
                col = 9
            elif doc_id < 1000:
                col = 10
            elif doc_id < 1100:
                col = 11
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
    col = None
    doc_id = int(doc_id)
    if doc_id < 100:
        col = 1
    elif doc_id < 200:
        col = 2
    elif doc_id < 300:
        col = 3
    elif doc_id < 400:
        col = 4
    elif doc_id < 500:
        col = 5
    elif doc_id < 600:
        col = 6
    elif doc_id < 700:
        col = 7
    elif doc_id < 800:
        col = 8
    elif doc_id < 900:
        col = 9
    elif doc_id < 1000:
        col = 10
    elif doc_id < 1100:
        col = 11
    text_file = os.path.dirname(__file__) + "/collection/" + str(col) + "/" + str(doc_id) +".txt"
    text = open(text_file).read()
    text = text.lower()

    context = {
        'doc_id': doc_id,
        'text': text,
    }
    return render(request, 'search/content.html', context)