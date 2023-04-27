# You can also adapt this script on your own summarization task. Pointers for this are left as comments.
import numpy as np

from numpy.linalg import norm
import re
import argparse
import json
import base64
import logging
import math
import os
import random
from pathlib import Path
import sys
import requests
import datasets
import nltk
import numpy as np
import configparser
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter
import transformers
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from flask_cors import CORS, cross_origin
from transformers.utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
import itertools
from elasticsearch7 import Elasticsearch
from sentence_transformers import SentenceTransformer, util

from flask import Flask, jsonify, request

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

configini = configparser.ConfigParser()
configini.read('config.ini')

eshost = configini['es']['host']
esport = configini['es']['port']

sparqlhost = configini['sparql']['host']
sparqlport = configini['sparql']['port']

labelsortlen = int(configini['candordering']['labelsort'])
embedsortlen = int(configini['candordering']['embedsort'])


es = Elasticsearch(host=eshost,port=int(esport))
entrankcounter = Counter()




model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)
#sparqlcache = json.loads(open('sparqlcache.json').read())
propdict = json.loads(open('en1.json').read())
goldrellabels = []
for k,v in propdict.items():
    goldrellabels.append([k,v])
sentence_embeddings = model.encode([x[1] for x in goldrellabels], convert_to_tensor=True)

entembedcache = {}

def getkgembedding(ent):
    if ent in entembedcache:
        return entembedcache[ent]
    entityurl = '<http://www.wikidata.org/entity/'+ent+'>'
    res = es.search(index="wikidataembedsindex01", body={"query":{"term":{"key":{"value":entityurl}}}})
    try:
        embedding = [float(x) for x in res['hits']['hits'][0]['_source']['embedding'][:10]]
        entembedcache[ent] = embedding
        return embedding
    except Exception as e:
        #print(ent,' entity embedding not found')
        return 10*[0.0]
    return 10*['0.0']


def getentlabel(ent):
    res = es.search(index="wikidataentitylabelindex02", body={"query":{"term":{"uri":{"value":ent}}}})
    try:
        return res['hits']['hits'][0]['_source']['wikidataLabel']
    except Exception as err:
        print(err)
        return ''

def getrellabel(rel):
    try:
        return propdict[rel]
    except Exception as err:
        return ''

def empty(r):
    if not r:
        return True
    if 'boolean' not in r:
        if 'results' in r:
            if 'bindings' in r['results']:
                if not r['results']['bindings']:
                    return True
                if {} in r['results']['bindings']:
                    return True
    return False


def fetchentitydetails(entityids,category):
    entities = []
    for entityid in entityids:
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entityid}&format=json&props=labels|descriptions|claims"
            print(url)
    
            # make a GET request to the API
            response = requests.get(url)
    
            # extract the label, description, and image URL from the response
            data = response.json()
            label = data["entities"][entityid]["labels"]["en"]["value"]
            description = data["entities"][entityid]["descriptions"]["en"]["value"]
            print(label)
            print(description)
            if category == "entity":
                try:
                    image_url_filename = data["entities"][entityid]["claims"]["P18"][0]["mainsnak"]["datavalue"]["value"]
                    image_url = f"https://commons.wikimedia.org/w/index.php?title=Special:Redirect/file/{image_url_filename}&width=100"
                    # download the image and encode it as a base64 string
                    image_data = requests.get(image_url).content
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                    # use the label, description, and image URL as needed
                    entities.append({"id":entityid, "label":label, "description":description, "image":encoded_image})
                except Exception as err:
                    print(err)
                    entities.append({"id":entityid, "label":label, "description":description})
            else:
                entities.append({"id":entityid, "label":label, "description":description})
        except Exception as err:
            print(err)
    return entities

def sparqlendpoint(query):
    try:
        url = 'http://%s:%d/api/endpoint/sparql'%(sparqlhost,int(sparqlport))
        query = '''PREFIX p: <http://www.wikidata.org/prop/> PREFIX pq: <http://www.wikidata.org/prop/qualifier/> PREFIX ps: <http://www.wikidata.org/prop/statement/>   PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wds: <http://www.wikidata.org/entity/statement/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> ''' + query
        #print(query)
        headers = {'Accept':'application/sparql-results+json'}
        r = requests.get(url, headers=headers, params={'format': 'json', 'query': query})
        json_format = r.json()
        #print(json_format)
        results = json_format
        return results
    except Exception as err:
        print(err)
        return ''


relcanddict = {}
def relcands(rellabel):
    if rellabel in relcanddict:
        return relcanddict[rellabel]
    results = []
    query = rellabel.strip()
    if not query:
        return []
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_results = torch.topk(cos_scores, k=3)

    #print("\n\n======================\n\n")
    #print("Query:", query)
    #print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        #print(goldrellabels[idx], "(Score: {:.4f})".format(score))
        results.append(goldrellabels[idx])
    relcanddict[rellabel] = results
    return results


def entcands(entlabel):
    esresults = es.search(index='wikidataentitylabelindex02',body={"query":{"match":{"wikidataLabel":entlabel}}},size=100)
    results = []
    try:
        for res in esresults['hits']['hits']:
            #print(entlabel, res['_source'])
            results.append([res['_source']['uri'],res['_source']['wikidataLabel']])
        return results
    except Exception as err:
        print(entlabel,err)
        return results


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    args = parser.parse_args()
    # Sanity checks

    return args


def infer(question):
    args = parse_args()

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    logger.setLevel(logging.INFO)

    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        #tokenizer.add_tokens(['{','}','select','where','?vr0','?vr1','?vr2','?vr3','?vr4','?vr5','?vr6'], special_tokens=True )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        #tokenizer.add_tokens(['{','}','select','where','?vr0','?vr1','?vr2','?vr3','?vr4','?vr5','?vr6'], special_tokens=True )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        ).to('cuda')

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""


    # Temporarily set max_target_length for training.
    max_target_length = 512
    padding = "do_not_pad"


    def preprocess_function():
        inputs = [question]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
        return model_inputs


    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        return preds

    model.eval()

    gen_kwargs = {
        "max_length": 512,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams
    }
    with torch.no_grad():
        input = preprocess_function()
        generated_tokens = model.generate(
            torch.tensor(input['input_ids']).to('cuda').long(),
            attention_mask=torch.tensor(input['attention_mask']).to('cuda').long(),
            **gen_kwargs,
        )
        generated_tokens = generated_tokens.cpu().numpy()

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print(decoded_preds)
        f = lambda A, n=3: [A[i:i+n] for i in range(0, len(A), args.num_beams)]
        beamed_preds = f(decoded_preds)
        print(beamed_preds)
        original_inputs = tokenizer.batch_decode(input["input_ids"], skip_special_tokens=True)
        print(original_inputs)
        nonempty = False
        beamoutputs = [] 
        for beams,original_input in zip(beamed_preds,original_inputs):
            beamitem = {}
            queryresult = []
            for beam in beams:
                pred = beam
                if nonempty:
                    break
                if 'order' in pred:
                     pred = pred.replace('?vr',' ?vr').replace(' where',' where {').replace(' kgembed>',' <kgembed>').replace('/kgembed>','</kgembed>').replace(')  @@',') { @@').replace(')  ?',') { ?').replace('order','} order')
                elif 'limit' in pred:
                     pred = pred.replace('?vr',' ?vr').replace(' where',' where {').replace(' kgembed>',' <kgembed>').replace('/kgembed>','</kgembed>').replace(')  @@',') { @@').replace(')  ?',') { ?').replace('limit','} limit')
                else:
                    pred = pred.replace('?vr',' ?vr').replace(' where',' where {').replace(' kgembed>',' <kgembed>').replace('/kgembed>','</kgembed>').replace(')  @@',') { @@').replace(')  ?',') { ?') +'}'
                print("preds:",pred,len(pred))
                beamitem['predicted_query_initial'] = pred
                kd = {}
                entlabels = re.findall( r'@@entbegin wd: \|\| (.*?) @@entend', pred)
                beamitem['entlabels'] = entlabels
                beamitem['candidateentities_presort'] = {}
                beamitem['candidateentities_postsort'] = {}
                for emblabel in entlabels:
                    #print("entlabel:",emblabel)
                    kgembed = re.findall(r'<kgembed> (.*?) </kgembed>',emblabel)
                    try:
                        kgembedding = [float(x) for x in kgembed[0].split()]
                        #print("kgemb:",kgembedding)
                    except Exception as err:
                        print(err)
                        continue
                    try:
                        label = re.findall(r'(.*?) <kgembed>',emblabel)[0]
                        ent_cands = entcands(label)
                        ent_cands_dots = [[x[0],x[1],np.dot(kgembedding,getkgembedding(x[0])[:len(kgembedding)])] for x in ent_cands]
                        ent_cands_dots_sorted = sorted(ent_cands_dots, key=lambda x: x[2], reverse = True)
                    except Exception as err:
                        print(err)
                        continue
                    beamitem['candidateentities_presort'][label] = fetchentitydetails([x[0] for x in ent_cands_dots[:labelsortlen]],"entity")
                    s = '''@@entbegin wd: || '''+label+''' <kgembed> '''+kgembed[0]+''' </kgembed> @@entend'''
                    kd[s] = [['wd:'+e[0],e[1]] for e in ent_cands_dots[:labelsortlen]]
                    kd[s] += [['wd:'+e[0],e[1]] for e in ent_cands_dots_sorted[:embedsortlen]]
                    beamitem['candidateentities_postsort'][label] = fetchentitydetails([x[0] for x in ent_cands_dots[:labelsortlen]+ent_cands_dots_sorted[:embedsortlen]],"entity")
                beamitem['rellabels'] = []
                beamitem['candidaterelations'] = {}
                for rel in ['p:','ps:','pq:','wdt:']:
                    rellabels = re.findall( r'@@relbegin '+rel+' \|\| (.*?) @@relend' ,pred)
                    beamitem['rellabels'] += rellabels
                    for label in rellabels:
                        print("rellabel:",label)
                        rel_cands = relcands(label)
                        beamitem['candidaterelations'][label] = [fetchentitydetails(x,"relation") for x in rel_cands]
                        s = '''@@relbegin '''+rel+''' || '''+label+''' @@relend'''
                        kd[s] = [[rel+r[0],r[1]] for r in rel_cands]
                iterlist = []
                for k,v in kd.items():
                    iterlist.append(list(range(len(v))))
                for tup in list(itertools.product(*iterlist)):
                    if  nonempty:
                        break
                    #print(tup)
                    tupcount = 0
                    m = pred
                    for k,v in kd.items():
                        #print("replace",k,"with",v[tup[tupcount]][0])
                        m = m.replace(k, v[tup[tupcount]][0])
                        tupcount += 1
                    #print("m = ",m)
                    queryresult = sparqlendpoint(m)
                    if empty(queryresult):
                        continue
                    else:
                        beamitem['predicted_query'] = m
                        beamitem['query_result'] = queryresult
                        print(tup)
                        print('querysparq:',m)
                        print("queryresul:",queryresult)
                        predents = re.findall(r'wd:(.*?) ',m)
                        predents = fetchentitydetails(predents,"entity")
                        predrels = []
                        predrels += re.findall(r'wdt:(.*?) ', m)
                        predrels += re.findall(r'p:(.*?) ', m)
                        predrels += re.findall(r'ps:(.*?) ', m)
                        predrels += re.findall(r'pq:(.*?) ', m)
                        predrels = fetchentitydetails(predrels,"relation")
                        beamitem['predicted_entities'] = predents
                        beamitem['predicted_relations'] = predrels
                        nonempty = True
                beamoutputs.append(beamitem) 
        return beamoutputs





app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/answer', methods=['POST'])
@cross_origin()
def answer():
    data = request.get_json()
    print(data)
    question = data['question']
    output_str = infer(question)
    return jsonify({'output': output_str})

if __name__ == '__main__':
    app.run(host="0.0.0.0")
