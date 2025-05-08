import sys,os,json

d = json.loads(open(sys.argv[1]).read())

f = open(sys.argv[2],'w')

for item in d:
    if not item:
        continue
    f.write(json.dumps({'text':item['question'], 'summary':item['masked_sparql'], 'gold_sparql':item['gold_sparql'], 'kgembeds':item['kgembeds']})+'\n')

f.close()
