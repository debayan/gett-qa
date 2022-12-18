
# GETT-QA: Graph Embedding based T2T Transformer for Knowledge Graph Question Answering



## Usage

### Knowledge Bases

First we must setup the knowledge bases required to replicate the results in the paper. For LC-QuAD 2.0 we use the NLIWOD verstion of Wikidata, while for SimpleQuestions-Wikidata, we use the Wiki4M version. For LC-QuAD 2.0, deploy the docker container hosted at https://hub.docker.com/r/qacompany/hdt-query-service and follow the instructions on the page to access the API locally. For SimpleQuestions-Wikidata, first setup an instance of Virtuoso DB, either via docker (https://hub.docker.com/r/tenforce/virtuoso/) or as a standalone service. Download the triples database from https://drive.google.com/file/d/14n36kPT3658jgf7aS0Z6cwB1_nsZLoIj/view?usp=sharing and place it in the data/ folder of virtuoso, and start the service.

### Labels and Embedding Indices for Elasticsearch

GETT-QA relies on Elasticsearch indices to perform label search for entity linking, and to fetch entity embeddings. Download the compressed entity label indices and un-compress them:

* For LC-QuAD 2.0 https://drive.google.com/file/d/165pRKzZv1T2xxRjzYzPV_89FNx-dL-mH/view?usp=sharing
* For SimpleQuestions-Wikidata https://drive.google.com/file/d/1qWSDVl_E9jD0sNTtQfPjC5J20y-XkQc2/view?usp=sharing

Download the corresponding Elasticsearch mapping json files too

* https://drive.google.com/file/d/186pqQPvLXkWHetPgS38HW0LbR7n6liQf/view?usp=sharing
* https://drive.google.com/file/d/1RQMZNCoPBFg_erZ9BAaZ_N8ktHEkm2Y3/view?usp=sharing

Install Elasticsearch 7.x version. It should listen on port 9200. 
Create an index, create mappings, and import data into the index

```
# curl -XPUT 'http://localhost:9200/wikidataentitylabelindex02'
# curl -XPUT 'http://localhost:9200/wikidataentitylabelindex02/_mapping' -d@wikidataentitylabelindex02.mapping.json.
# curl -XPUT 'http://localhost:9200/wiki4mlabels1'
# curl -XPUT 'http://localhost:9200/wiki4mlabels1/_mapping' -d@wiki4mlabels1.mapping.json
```
To import data, install the tool named `elasticdump ( `npm install elasticdump@6.33.4` )` 

```
# elasticdump --limit=10000 --input=wikidataentitylabelindex02.json --output=http://localhost:9200/wikidataentitylabelindex02 --type=data
# elasticdump --limit=10000 --input=wiki4mlabels1.json --output=http://localhost:9200/wiki4mlabels1 --type=data
```

### Code

We are now ready to run the code. First, create a `pip` virtual environment

```
# python3.10 -m venv .
# source bin/activate
# pip install -r requirements.txt
```
Once the python dependencies are installed, we may run the code.

### Reproduce Results

To directly reproduce the results in the paper one may download the pre-trained model weights.
* LC-QuAD 2.0 https://drive.google.com/file/d/1gZ42bWD4NY25CZLNX7rfm2kwqZPfxuV9/view?usp=sharing
* SimpleQuestions-Wikidata https://drive.google.com/file/d/1B_MwC9Jh505sBrukMsMi9mSXouQ9rnLq/view?usp=sharing

Uncompress them and move them into the corresponding folders for the respective datasets.

For LC-QuAD 2.0, 
```
# cd lcquad2/
# vi config.ini
```
Here edit the server host and port details of Elasticsearch and SPARQL endpoints. 

To run the code with the model:

```
# CUDA_VISIBLE_DEVICES=0 python -u t5_infer_kgembed.py --model_name_or_path models/epoch_99/ --validation_file data/test_kgembed_jsonlines_uniqorn_4921_1.json --source_prefix "summarize: " --per_device_eval_batch_size=50 --num_beams=3 | tee logs/logeval1.txt**
```
Similarly for SimpleQuestions-Wikidata in its corresponding folder `simplequestions/`. Note that the SPARQL endpoint details will be virtuoso based and different.

```
# CUDA_VISIBLE_DEVICES=2 python -u t5_infer_kgembed.py --model_name_or_path models/epoch_99/ --validation_file data/test_wiki4m_jsonlines_1.json --source_prefix "summarize: " --per_device_eval_batch_size=20 --num_beams=3 | tee logs/logeval1.txt
```
In `config.ini`change the values of `labelsort` and `embedsort` from 3,3 to 6,0 to produce results for label-based sorting only as reported in the paper.

## Train

To train, for example, LC-QuAD 2.0:

```
# CUDA_VISIBLE_DEVICES=0 python -u train.py --model_name_or_path t5-base --train_file data/train_kgembed_jsonlines_1.json --validation_file data/dev_kgembed_jsonlines_1.json --source_prefix "summarize: " --output_dir modelx --per_device_train_batch_size=4 --per_device_eval_batch_size=8 --learning_rate=1e-4 --num_train_epochs=100
```

For SimpleQuestions-Wikidata:

```
# CUDA_VISIBLE_DEVICES=2 python -u train.py --model_name_or_path t5-base --train_file data/train_jsonlines_1.json --validation_file data/dev_jsonlines_1.json --source_prefix "summarize: " --output_dir modelx --per_device_train_batch_size=4 --per_device_eval_batch_size=8 --learning_rate=1e-4 --num_train_epochs=100
```

##  Generating Training Files
For the training process, from the original datasets, we need to generate modified SPARQL queries where the entity and relation IDs are replaced by labels, and KG embedding snippets are added to the queries.  Usually there is no need to perform this step as these specially formed SPARQL queries are pre-generated and already present in the current code base. However, should one need to re-generate these files, here are the instructions.

For LC-QuAD 2.0:

```
# python noisyques2entlabel_kgembeddings.py dataset/train.json x.json
# python tojsonlines.py x.json data/train_kgembed_jsonlines_1.json
# python noisyques2entlabel_kgembeddings.py dataset/test_kgembedding_uniqorn_4921_1.json x.json
# python tojsonlines.py data/test_kgembed_jsonlines_uniqorn_4921_1.json
```
For SimpleQuestions-Wikidata:

```
# python noisyques2entlabel_kgembeddings.py datasets/annotated_wd_data_train.txt x.json
# python tojsonlines.py x.json data/train_jsonlines_1.json
# python noisyques2entlabel_kgembeddings.py datasets/annotated_wd_data_test_wiki4m.txt x.json
# python tojsonlines.py x.json data/test_wiki4m_jsonlines_1.json
```


