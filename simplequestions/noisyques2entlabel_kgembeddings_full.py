import sys,os,json,re
from copy import deepcopy
from elasticsearch7 import Elasticsearch
import configparser


d = [x.strip() for x in open(sys.argv[1]).readlines()]
simplequestions = []
for item in d:
    s,p,o,q = item.split('\t')
    simplequestions.append({'s':s,'p':p,'o':o,'q':q})

print(len(simplequestions))

configini = configparser.ConfigParser()
configini.read('config.ini')

eshost = configini['es']['host']
esport = configini['es']['port']

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
        embedding = res['hits']['hits'][0]['_source']['embedding']
        entembedcache[ent] = embedding
        return embedding
    except Exception as e:
        print(ent,' entity embedding not found')
        return 200*[0.0]
    return 200*[0.0]

for idx,item in enumerate(simplequestions):
    citem = {}
    s,p,o,q = item['s'],item['p'],item['o'],item['q']
    wikisparql = "select ?vr0 where { wd:%s wdt:%s ?vr0 }"%(s,p)
    citem['gold_sparql'] = wikisparql
    print(wikisparql)
    unit = {}
    unit['uid'] = idx
    unit['question'] = q
    #unit['paraphrased_question'] = item['paraphrased_question']
    ents = [s]
    rels = [p]
    entlabelarr = []
    kgembeds = []
    for ent in ents:
        try:
            label = getlabel(ent)
            if not label:
                continue
            kgembed = getkgembedding(ent)
            kgembeds.append(kgembed)
            wikisparql = wikisparql.replace(' wd:'+ent, ' @@ENTBEGIN wd: || '+label+ ' @@ENTEND')
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
            rellabelarr.append(label)
        except Exception as err:
            print(err)
            continue
    #rellabelarr.sort()
    citem['question'] = q
    citem['masked_sparql'] = wikisparql.lower()
    citem['kgembeds'] = kgembeds 
    print(idx, s,p,o,q)
    print(citem['masked_sparql'])
    arr.append(citem)

f = open(sys.argv[2],'w')
f.write(json.dumps(arr, indent=4))
f.close()
