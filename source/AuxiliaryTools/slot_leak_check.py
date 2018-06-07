# coding:utf-8

import json
import os
import re
LOG_DIR = '../../log/'
CONFIG_PATH = '../../config.json'
with open(CONFIG_PATH, 'r') as reader:
    CONFIG = json.load(reader)

test_slot_file_path = CONFIG['path']["ClusteringResult"] + 'test_atis_labeled_intent-slot1.json'
origin_train_file_path = CONFIG['path']['RawData']['atis'] + 'atis_train'

for split_rate in [129, 4478]:
# for split_rate in CONFIG['experiment']['train_set_split_rate']:
    target_generation_file_path = CONFIG['path']['OnmtData'] + 'Result/' + f'atis_labeled_intent-slot{split_rate}_pred_refilled.txt'

    all_test_slot_value_word = set()
    test_only_value_word = set()
    new_slot_count = 0
    ori_slot_count = 0

    with open(test_slot_file_path, 'r') as reader:
        test_json_data = json.load(reader)
    for data_item_lst in test_json_data.values():
        for data_item in data_item_lst:
            for slot_value_word_lst in data_item['slot_value_lst']:
                all_test_slot_value_word = all_test_slot_value_word | set(slot_value_word_lst)

    with open(origin_train_file_path, 'r') as reader:
        all_origin_text = reader.read().lower()
    with open(target_generation_file_path, 'r') as reader:
        print(target_generation_file_path)
        all_generated_text = reader.read()
    for word in all_test_slot_value_word:
        if word not in all_origin_text:
            print(word)
            test_only_value_word.add(word)
    print(len(test_only_value_word), 'new word in total', len(all_test_slot_value_word) - len(test_only_value_word), 'slot words in total')
    for word in test_only_value_word:
        if word in all_generated_text:
            print(word, 'leaked!!!!!')
