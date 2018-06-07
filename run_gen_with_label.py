# coding:utf-8
import json
import os
import copy
import subprocess
import argparse
from source.AuxiliaryTools.nlp_tool import low_case_tokenizer, sentence_edit_distance
from source.ReFilling.re_filling import re_filling
import math
from collections import Counter
import random
from itertools import combinations
from set_config import refresh_config_file

# ============ Description ==========
# append source id in the end, and use tree bank tokenize

# ============ Args Process ==========
# General function
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default='atis_former', help="choose task: atis_former")
parser.add_argument("-gd", "--gen_data", help="generate data from cluster result", action="store_true")
parser.add_argument("-cd", "--cook_data", help="cook data for ONMT", action="store_true")
parser.add_argument("-trn", "--train", help="run train part", action="store_true")
parser.add_argument("-tst", "--test", help="run test part", action="store_true")
parser.add_argument("-gcd", "--gen_conll_data", help="convert generated data to conll format", action="store_true")
parser.add_argument("-f", "--full", help="run all part", action="store_true")

# Deep Customize
parser.add_argument("-gpu", "--gpu_id", type=int, default=0, help='input gpu id for current task, -1 to not use gpu')
parser.add_argument('--config', default='./config.json', help="specific a config file by path")
args = parser.parse_args()

# ============ Refresh Config ==========
refresh_config_file(args.config)

# ============ Settings ==========
TASK_NAME = args.task
GEN_DATA = args.gen_data  # or args.full
COOK_DATA = args.cook_data or args.full
RUN_TRAIN = args.train or args.full
RUN_TEST = args.test or args.full
GEN_CONLL = args.gen_conll_data or args.full
with open(args.config, 'r') as con_f:
    CONFIG = json.load(con_f)


def get_file_tail(task_name, split_rate):
    return f"{task_name}-{split_rate}"



def generate_data(input_file_path, output_src_file_path, output_tgt_file_path):
    with open(input_file_path, 'r') as reader, \
            open(output_src_file_path, 'w') as src_writer, \
            open(output_tgt_file_path, 'w') as tgt_writer:
        json_data = json.load(reader)
        all_pairs = []
        for cluster_item in json_data.values():
            for turn in cluster_item:
                temp_source_word_lst = turn['user_word_lst']
                temp_target_word_lst = []
                for i in range(len(temp_source_word_lst)):
                    temp_target_word_lst.append(turn['user_word_lst'][i] + '<' + turn['label_lst'][i] + '>')

                all_pairs.append([temp_source_word_lst, temp_target_word_lst])
        print('Input:\n%s\nOutput to:\n%s\n%s\n' % (input_file_path,output_src_file_path,output_tgt_file_path))
        for ind, pair in enumerate(all_pairs):
            src_writer.write(' '.join(pair[0]) + '\n')
            tgt_writer.write(' '.join(pair[1]) + '\n')
            if ind % 1000 == 0:
                print(ind, 'pairs writen')
        print(len(all_pairs), 'pairs in total.')


def convert_to_conll(origin_file, input_file_path, output_file_path):
    with open(origin_file, 'r')as ori_file, open(input_file_path, 'r') as reader, open(output_file_path, 'w') as writer:
        appeared_line_set = set()
        json_data = json.load(ori_file)
        for cluster_item in json_data.values():
            for turn in cluster_item:
                temp_source_word_lst = turn['user_word_lst']
                temp_target_word_lst = []
                for i in range(len(temp_source_word_lst)):
                    temp_target_word_lst.append(turn['user_word_lst'][i] + '<' + turn['label_lst'][i] + '>')
                appeared_line_set.add(' '.join(temp_target_word_lst))

        for line in reader:
            appeared_line_set.add(line.replace('\n', ''))

        for line in appeared_line_set:
            word_label_lst = line.split()
            for word_label in word_label_lst:
                # print('debug', word_label)
                # print('debug', word_label_lst)
                if '<unk>' in word_label:
                    word = 'unk'
                    label = word_label.split('><')[1].replace('>', '')
                else:
                    word, label = word_label.split('<')
                    label = label.replace('>', '')
                writer.write('%s\t%s\n' % (word, label))
            writer.write('\n')


def dress_param_with_config(param_config, replace_table):
    # replace "data_mark, data_dir, result_dir" slots in param
    ret = copy.deepcopy(param_config)
    for key in ret:
        param_str = " ".join(ret[key])
        for slot_name in replace_table:
            param_str = param_str.replace(slot_name, replace_table[slot_name])
        param_lst = param_str.split()
        ret[key] = param_lst
    return ret


def call_onmt(task_name, param):
    print('========================== Call Onmt for: %s ==========================' % task_name)
    print('==========================       Param       ==========================\n%s' % ' '.join(param))
    print('==========================  Open-NMT Output  ========================== \n')
    proc = subprocess.Popen(param, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open('log/' + task_name + 'log', 'w') as writer:
        has_error = False
        for line in proc.stdout.readlines():
            print(line)
            writer.write(line.rstrip().decode("utf-8") + '\n')
            if b'error' in line.lower() and b'check_error' not in line.lower():
                has_error = True
        if has_error:
            raise RuntimeError


def main():
    all_split_rate = CONFIG['experiment']['train_set_split_rate']

    if GEN_DATA:
        clustering_results_dir = CONFIG['path']['ClusteringResult']
        onmt_source_dir = CONFIG['path']['OnmtData'] + 'SourceData/'
        # gen train data
        for split_rate in all_split_rate:
            input_file_path =  clustering_results_dir + f'train_atis_labeled_intent-slot{split_rate}.json'
            output_src_file_path =  onmt_source_dir + 'train_' + get_file_tail(TASK_NAME, split_rate) + '_src.txt'
            output_tgt_file_path =  onmt_source_dir + 'train_' + get_file_tail(TASK_NAME, split_rate) + '_tgt.txt'
            generate_data(input_file_path,output_src_file_path, output_tgt_file_path)
        # gen dev data
        input_file_path_for_dev = clustering_results_dir + 'dev_atis_labeled_intent-slot1.json'
        output_src_file_path_for_dev = onmt_source_dir + 'dev_' + get_file_tail(TASK_NAME, 1) + '_src.txt'
        output_tgt_file_path_for_dev = onmt_source_dir + 'dev_' + get_file_tail(TASK_NAME, 1) + '_tgt.txt'
        generate_data(input_file_path_for_dev, output_src_file_path_for_dev, output_tgt_file_path_for_dev)
        # gen test data
        input_file_path_for_test = clustering_results_dir + 'dev_atis_labeled_intent-slot1.json'
        output_src_file_path_for_test = onmt_source_dir + 'test_' + get_file_tail(TASK_NAME, 1) + '_src.txt'
        output_tgt_file_path_for_test = onmt_source_dir + 'test_' + get_file_tail(TASK_NAME, 1) + '_tgt.txt'
        generate_data(input_file_path_for_test, output_src_file_path_for_test, output_tgt_file_path_for_test)

    for split_rate in all_split_rate:
        # Customize these parameters for OpenNMT tool

        train_file_tail = get_file_tail(task_name=TASK_NAME, split_rate=split_rate)
        dev_file_tail = get_file_tail(task_name=TASK_NAME, split_rate=1)
        param_replace_table = {
            '<DATA_DIR>': CONFIG['path']['OnmtData'] + 'SourceData',
            '<RESULT_DIR>': CONFIG['path']['OnmtData'] + 'Result',
            '<TRAIN_FILE_TAIL>': train_file_tail,
            '<DEV_FILE_TAIL>': dev_file_tail,
            '<GPU>': '' if args.gpu_id < 0 else ('-gpu %d' % args.gpu_id),
        }
        print('Debug', param_replace_table)
        param_config = dress_param_with_config(CONFIG['gen_with_label'], param_replace_table)
        if COOK_DATA:
            # to get word embedding and dict
            call_onmt('prepare_data: ' + train_file_tail, param_config['prepare_data'])
        if RUN_TRAIN:
            call_onmt('train: ' + train_file_tail, param_config['train'])
        if RUN_TEST:
            call_onmt('test: ' + train_file_tail, param_config['test'])
        if GEN_CONLL:
            origin_file_path = CONFIG['path']['ClusteringResult'] + f'train_atis_labeled_intent-slot{split_rate}.json'
            input_file_path = CONFIG['path']['OnmtData'] + 'Result/' + train_file_tail + '_pred.txt'
            output_file_path = CONFIG['path']['Evaluate'] + 'SlotFilling/Source/' + 'extend_train_' + train_file_tail + '.conll'
            convert_to_conll(origin_file_path, input_file_path, output_file_path)


if __name__ == "__main__":
    main()
    print('Warn! Turn on NOISE_TRANLATE in onmt\'s translator.py')
