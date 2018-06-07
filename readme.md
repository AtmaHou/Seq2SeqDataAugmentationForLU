# Sequence-to-sequence Data Augmentation for Dialogue Language Understanding
## 用户输入多样性拓展生成
#### Author: Atma
#### Update: 2018/6/7

# Introduction

This repo is code for the COLING 2018 paper: Sequence-to-sequence Data Augmentation for Dialogue Language Understanding.

## Get started
The following steps show code usage for the ATIS dataset.

- Step1: Clustering Sentences

    python3 run_clustering.py -d atis

-- Tips;
    To remove clustering effects for baseline setting i.e. cluster all data into one class:


- Step2: Prepare data

    python3 run_onmt_generation.py  -gd

-- Tips:
    There are some alternatives for baseline setting:

    No clustering, Full connect , no index
	python3 run_onmt_generation.py -gd -pm circle -ni -nc

	Full connect , no index
	python3 run_onmt_generation.py  -gd -pm full_connect -ni

	Diverse connect, no index
	python3 run_onmt_generation.py  -gd -ni

	Diverse connect, no filtering
    python3 run_onmt_generation.py  -gd -fr 1

- Step3: Seq2Seq Generation

    python3 run_onmt_generation.py -t atis_labeled -f

-- Tips:
    Again, alternatives for baseline:

    No clustering, Full connect , no index
    python3 run_onmt_generation.py -t atis_labeled -f -pm circle -ni -nc

    Full connect , no index  ===> running
    python3 run_onmt_generation.py -t atis_labeled -f -pm full_connect -ni

    Diverse connect, no index
    python3 run_onmt_generation.py -t atis_labeled -f -ni

    Diverse connect, no filtering
    CUDA_VISIBLE_DEVICES="1" python3 run_onmt_generation.py  -t atis_labeled -f -fr 1

- Step4: Surface Realization

    python3 run_onmt_generation.py -t atis_labeled -rf

-- Tips:

    For surface realization only baseline:
    python3 run_thesaurus.py -t atis_labeled -rf

- Step5: Generate Conll Format Data

    python3 run_slot_filling_evaluation.py -t atis_labeled -gd xiaoming -cd

-- Tips:
   For surface realization only baseline:
   python3 run_slot_filling_evaluation.py -t atis_labeled -gd xiaoming -cd -rfo


# Notice

As the slot-filling used by our work is simply Bi-LSTM and our augmentation method suit for all slot-filling algorithm,
we only release the seq2seq argumentation part and CONLL format data generation part.

You can add your own slot filling algorithm.
