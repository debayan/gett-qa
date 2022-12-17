import sys,os,json,re
from copy import deepcopy
from elasticsearch7 import Elasticsearch
import configparser


configini = configparser.ConfigParser()
configini.read('config.ini')

eshost = configini['es']['host']
esport = configini['es']['port']


d = json.loads(open(sys.argv[1]).read())
props = json.loads(open('en1.json').read())
arr = []
es = Elasticsearch(host=eshost,port=int(esport))

def getlabel(ent):
    results = es.search(index='wikidataentitylabelindex02',body={"query":{"term":{"uri":{"value":ent}}}})
    try:
        for res in results['hits']['hits']:
            return res['_source']['wikidataLabel']
    except Exception as err:
        print(results)
        print(ent,err)
        return ''

entembedcache = {}

def getkgembedding(ent):
    if ent in entembedcache:
        return entembedcache[ent]
    entityurl = '<http://www.wikidata.org/entity/'+ent+'>'
    res = es.search(index="wikidataembedsindex01", body={"query":{"term":{"key":{"value":entityurl}}}})
    try:
        embedding = ' '.join([str(x) for x in res['hits']['hits'][0]['_source']['embedding']][:10])
        entembedcache[ent] = embedding
        return embedding
    except Exception as e:
        print(ent,' entity embedding not found')
        return ' '.join(10*['0.0'])
    return ' '.join(10*['0.0'])

for item in d:
    citem = deepcopy(item)
    wikisparql = item['sparql_wikidata'].replace('(',' ( ').replace(')',' ) ').replace('{',' { ').replace('}',' } ')
    print(wikisparql)
    unit = {}
    unit['uid'] = item['uid']
    unit['question'] = item['question']
    if not item['question']:
        continue
    #unit['paraphrased_question'] = item['paraphrased_question']
    ents = re.findall( r'wd:(.*?) ',wikisparql)
    rels = re.findall( r'wdt:(.*?) ',wikisparql)
    rels += re.findall( r'p:(.*?) ',wikisparql)
    rels += re.findall( r'ps:(.*?) ',wikisparql)
    rels += re.findall( r'pq:(.*?) ',wikisparql)
    entlabelarr = []
    for ent in ents:
        try:
            label = getlabel(ent)
            if not label:
                continue
            kgembed = getkgembedding(ent)
            wikisparql = wikisparql.replace(' wd:'+ent, ' @@ENTBEGIN wd: || '+label+  ' <kgembed> '+kgembed + ' </kgembed> @@ENTEND')
            #entlabelarr.append(label)
        except Exception as err:
            print(err)
            sys.exit(1)
            continue
    #entlabelarr.sort()
    rellabelarr = []
    for rel in rels:
        try:
            label = props[rel]
            if not label:
                continue
            wikisparql = wikisparql.replace(' wdt:'+rel, ' @@RELBEGIN wdt: || '+label+' @@RELEND')
            wikisparql = wikisparql.replace(' p:'+rel, ' @@RELBEGIN p: || '+label+' @@RELEND')
            wikisparql = wikisparql.replace(' ps:'+rel, ' @@RELBEGIN ps: || '+label+' @@RELEND')
            wikisparql = wikisparql.replace(' pq:'+rel, ' @@RELBEGIN pq: || '+label+' @@RELEND')
            rellabelarr.append(label)
        except Exception as err:
            print(err)
            continue
    #rellabelarr.sort()
    print(wikisparql)
    newvars = ['?vr0','?vr1','?vr2','?vr3','?vr4','?vr5']
    sparql_split = wikisparql.split()
    variables = set([x for x in sparql_split if x[0] == '?'])
    print(variables)
    for idx,var in enumerate(sorted(variables)):
        wikisparql = wikisparql.replace(var,newvars[idx])
    print(wikisparql)
    citem['masked_sparql'] = wikisparql.lower()
    arr.append(citem)

f = open(sys.argv[2],'w')
f.write(json.dumps(arr, indent=4))
f.close()
