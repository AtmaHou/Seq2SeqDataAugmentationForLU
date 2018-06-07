# coding: utf-8
from source.AuxiliaryTools.nlp_tool import low_case_tokenizer
from itertools import combinations
import pickle
import json
import os

SOS_token = 0
EOS_token = 1


class WordTable:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def utterance_tokenize(clustered_items):
    """
    tokenize the utterance with nlp_tool.py
    :param clustered_items: List, dialogue pairs with same user intents
    :return:
    """
    for ind, item in enumerate(clustered_items):
        clustered_items[ind]['user_temp'] = low_case_tokenizer(item['user_temp'])
        clustered_items[ind]['agent_temp'] = low_case_tokenizer(item['agent_temp'])
    return clustered_items


def utterance_filter(clustered_items):
    """
     remove bad utterance and duplicate
    :param clustered_items: List, dialogue pairs with same user intents
    :return:
    """
    ret = {}
    for item in clustered_items:
        user_temp = item['user_temp']
        user_say = item['user_say']
        if len(user_temp) <= 7 and ('thank' in user_temp or 'thanks' in user_temp):
            # abandon 'mean less' utterance
            continue
        if (user_say not in ret) or (item['agent_say'] and len(ret[user_say]['agent_say']) < len(item['agent_say'])):
            # when there is a duplicate, keep the one with higher reply quality
            ret[user_say] = item
    return ret.values()


def data_stat(all_data):
    return []


def cluster_to_pairs(cluster_item):
    """
    Construct source, target pairs from cluster
    :param cluster_item: List, dialogue pairs with same user intents
    :return:
    """
    all_utterance = [item["user_temp"] for item in cluster_item]
    all_combination = []
    for comb in combinations(all_utterance, 2):
        all_combination.append(comb)  # use combination to avoid self to self pairs
        all_combination.append(comb[::-1])  # to get reverse of it
    return all_combination


def prepare_data(config):
    """
    This prepare the data with following steps
    Step1: Tokenizing
    Step2: Filter the meaningless data
    Step3: Remove duplicate
    Step4: Build word table
    Step5: Generate pair wise Target & Source
    :param config: config data
    :return:
    """
    raw_data_file_lst = os.listdir(config['path']['ClusteringResult'])
    for f in raw_data_file_lst:
        with open(config['path']['ClusteringResult'] + f, 'r') as reader:
            json_data = json.load(reader)
            all_src_tgt_pairs = []
            file_mark = f.replace('.json', '')
            for key in json_data:
                # print('======== debug ========', type(json_data), f)
                tokenized_cluster = utterance_tokenize(json_data[key])
                filtered_cluster = utterance_filter(tokenized_cluster)
                all_src_tgt_pairs.extend(cluster_to_pairs(filtered_cluster))

            # === build word table ===
            current_word_table = WordTable(name=file_mark)
            for pairs in all_src_tgt_pairs:
                current_word_table.add_sentence(pairs[0])
                # add the second sent is useless because of the reverse opt
                # current_word_table.add_sentence(pairs[1])

            # === Export to file, avoid execute everytime ===
            output_dir = config['path']['GenerationResult']
            with open(output_dir + file_mark + '_pairs.json', 'w') as writer:
                json.dump(all_src_tgt_pairs, writer)
            with open(output_dir + file_mark + '_word-table.pickle', 'wb') as writer:
                pickle.dump(current_word_table, writer)
