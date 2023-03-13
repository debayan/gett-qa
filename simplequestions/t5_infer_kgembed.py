# You can also adapt this script on your own summarization task. Pointers for this are left as comments.
import numpy as np
from numpy.linalg import norm
import re
import argparse
import json
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
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import configparser
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
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
from transformers.utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
import itertools
from elasticsearch7 import Elasticsearch
from sentence_transformers import SentenceTransformer, util
from collections import Counter
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
        embedding = [float(x) for x in res['hits']['hits'][0]['_source']['embedding'][:20]]
        entembedcache[ent] = embedding
        return embedding
    except Exception as e:
        #print(ent,' entity embedding not found')
        return 20*[0.0]
    return 20*['0.0']


def getentlabel(ent):
    res = es.search(index="wiki4mlabels1", body={"query":{"term":{"uri":{"value":ent}}}})
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

def tpfpfn2(goldres,queryres):
    gold = []
    query = []
    tp = 0
    fp = 0
    fn = 0
    if 'results' in goldres:
        if 'bindings' in goldres['results']:
            if goldres['results']['bindings']:
                for x in goldres['results']['bindings']:
                    value = list(x.values())[0]['value']
                    gold.append(value)
    else:
        gold.append(goldres)  #For questions that do not have bindings array, like boolean, count, add the entire (singular) result to be compared, since tp fp fn does not make much sense here
    if 'results' in queryres:
        if 'bindings' in queryres['results']:
            if queryres['results']['bindings']:
                for x in queryres['results']['bindings']:
                    value = list(x.values())[0]['value']
                    query.append(value)
    else:
        query.append(queryres)
    for g in gold:
        if g in query:
            tp += 1
    for g in gold:
        if g not in query:
            fn += 1
    for q in query:
        if q not in gold:
            fp += 1
    return tp,fp,fn
    

def tpfpfn(goldres, queryres):
    if goldres == queryres:
        return 1,0,0 #tp,fp,fn
    if not queryres:
        return 0,0,1
    if goldres != queryres:
        return 0,1,1
    print("LOGIC ERROR")
    sys.exit(1)


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
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=512,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
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
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
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
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )

    args = parser.parse_args()

    # Sanity checks
    if  args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def main():
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
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.validation_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    gold_sparqls = [json.loads(line)['gold_sparql'] for line in open(args.validation_file).readlines()]
    gold_mask_sparqls = [json.loads(line)['summary'] for line in open(args.validation_file).readlines()]
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
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

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["validation"].column_names

    # Get the column names for input/target.
    print(column_names)
    text_column = column_names[0]
    summary_column = column_names[1]
    gold_column = column_names[2]

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        #gold_sparqls = tokenizer(examples[gold_column], max_length=args.max_target_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        #gold_sparqls["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in gold_sparqls["input_ids"]]
            

        model_inputs["labels"] = labels["input_ids"]
        #model_inputs["gold_sparql"] = gold_sparqls["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        #preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        #labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams
    }
    samples_seen = 0
    match = 0
    count = 0
    nonempty_count = 0
    dot_sum = 0
    dot_count = 0
    cos_sum = 0
    cos_count = 0
    tp = 0
    fp = 0
    fn = 0
    entmatch = 0
    relmatch = 0
    macrof1sum = 0
    macroprecisionsum = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"].to('cuda'),
                attention_mask=batch["attention_mask"].to('cuda'),
                **gen_kwargs,
            )
            #generated_tokens = accelerator.pad_across_processes(
            #    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            #)

            #generated_tokens = accelerator.gather((generated_tokens))
            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            #print(decoded_preds)
            f = lambda A, n=3: [A[i:i+n] for i in range(0, len(A), args.num_beams)]
            beamed_preds = f(decoded_preds)
            #print(beamed_preds)
            #decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            original_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            print(original_inputs)
            #gold_sparqls = tokenizer.batch_decode(batch["gold_sparql"], skip_special_tokens=True)

            #decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            for beams,original_input in zip(beamed_preds,original_inputs):
                gold_sparql = gold_sparqls[count]
                newvars = ['?vr0','?vr1','?vr2','?vr3','?vr4','?vr5']
                sparql_split = gold_sparql.split()
                variables = set([x for x in sparql_split if x[0] == '?'])
                for idx,var in enumerate(sorted(variables)):
                    gold_sparql = gold_sparql.replace(var,newvars[idx])
                gold_mask_sparql = gold_mask_sparqls[count]
                count += 1
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("match = %d/%d %f %%"%(match,nonempty_count,match/(nonempty_count+0.0001)))
                print("entmatch = %d/%d %f"%(entmatch,nonempty_count,entmatch/(nonempty_count+0.0001)))
                print("relmatch = %d/%d %f"%(relmatch,nonempty_count,relmatch/(nonempty_count+0.0001)))
                print("tp %d fp %d fn %d"%(tp,fp,fn))
                print("micro F1 = %f"%(tp/(tp+0.5*(fp+fn+0.000001))))
                print("micro P@1 %f"%(tp/(tp+fp+0.000001)))
                print("macro F1 = %f"%(macrof1sum/(nonempty_count+0.000001)))
                print("macro P@1 %f"%(macroprecisionsum/(nonempty_count+0.000001)))
                print("input:",original_input)
                print("gold sparql:", gold_sparql)
                print("gold mask sparql:",gold_mask_sparql)
                print("entrankcounter:",entrankcounter)
                nonempty = False
                matchfound = False
                queryresult = []
                goldresult = sparqlendpoint(gold_sparql)
                if empty(goldresult):
                    print("EMPTY GOLD, SKIPPING")
                    print("===================================")
                    #sys.exit(1)
                    continue
                nonempty_count += 1
                for beam in beams:
                    pred = beam
                    if nonempty or matchfound:
                        break
                    if 'order' in pred:
                         pred = pred.replace('?vr',' ?vr').replace(' where',' where {').replace(' kgembed>',' <kgembed>').replace('/kgembed>','</kgembed>').replace(')  @@',') { @@').replace(')  ?',') { ?').replace('order','} order')
                    elif 'limit' in pred:
                         pred = pred.replace('?vr',' ?vr').replace(' where',' where {').replace(' kgembed>',' <kgembed>').replace('/kgembed>','</kgembed>').replace(')  @@',') { @@').replace(')  ?',') { ?').replace('limit','} limit')
                    else:
                        pred = pred.replace('?vr',' ?vr').replace(' where',' where {').replace(' kgembed>',' <kgembed>').replace('/kgembed>','</kgembed>').replace(')  @@',') { @@').replace(')  ?',') { ?') +'}'
                    print("preds:",pred,len(pred))
                    kd = {}
                    entlabels = re.findall( r'@@entbegin wd: \|\| (.*?) @@entend', pred)
                    ent_cands = None
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
                        s = '''@@entbegin wd: || '''+label+''' <kgembed> '''+kgembed[0]+''' </kgembed> @@entend'''
                        kd[s] = [['wd:'+e[0],e[1]] for e in ent_cands_dots[:labelsortlen]]
                        kd[s] += [['wd:'+e[0],e[1]] for e in ent_cands_dots_sorted[:embedsortlen]]
                    for rel in ['wdt:']:
                        rellabels = re.findall( r'@@relbegin '+rel+' \|\| (.*?) @@relend' ,pred)
                        for label in rellabels:
                            print("rellabel:",label)
                            rel_cands = relcands(label)
                            s = '''@@relbegin '''+rel+''' || '''+label+''' @@relend'''
                            kd[s] = [[rel+r[0],r[1]] for r in rel_cands]
                    iterlist = []
                    for k,v in kd.items():
                        iterlist.append(list(range(len(v))))
                    for tup in list(itertools.product(*iterlist)):
                        if matchfound or nonempty:
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
                            print(tup)
                            print('goldsparql:',gold_sparql)
                            print('querysparq:',m)
                            print("goldresult:",goldresult)
                            print("queryresul:",queryresult)
                            goldents = re.findall(r'wd:(.*?) ',gold_sparql)
                            goldrels = re.findall(r'wdt:(.*?)', gold_sparql)
                            predents = re.findall(r'wd:(.*?) ',m)
                            predrels = re.findall(r'wdt:(.*?)', m)
                            if goldents == predents:
                                entmatch += 1
                            if goldrels == predrels:
                                relmatch += 1
                            if queryresult == goldresult:
                                print("MATCH")
                                for idx,entcand in enumerate(ent_cands):
                                    if predents[0] == entcand[0]:
                                        entrankcounter[idx+1] += 1
                                match += 1
                                matchfound = True
                            nonempty = True
                _tp,_fp,_fn = tpfpfn2(goldresult,queryresult)
                macrof1sum += _tp/(_tp+0.5*(_fp+_fn)+0.000001)
                macroprecisionsum += _tp/(_tp+_fp+0.0000001)
                tp += _tp
                fp += _fp
                fn += _fn



if __name__ == '__main__':
    main()
