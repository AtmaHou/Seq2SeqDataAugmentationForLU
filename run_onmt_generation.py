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
parser.add_argument("-t", "--task", type=str, default='atis_labeled', help="choose task: atis_labeled, atis_baseline, navigate_labeled, schedule_labeled, weather_labeled, navigate, schedule, weather")
parser.add_argument("-gd", "--gen_data", help="generate data from cluster result", action="store_true")
parser.add_argument("-cd", "--cook_data", help="cook data for ONMT", action="store_true")
parser.add_argument("-trn", "--train", help="run train part", action="store_true")
parser.add_argument("-tst", "--test", help="run test part", action="store_true")
parser.add_argument("-rf", "--refill", help="run surface realization", action="store_true")
parser.add_argument("-f", "--full", help="run all part", action="store_true")

# Deep Customize
parser.add_argument("-pm", "--pair_mode", type=str, default='diverse_connect', help='choose mode: "full_connect", "circle", "diverse_connect","random"')
parser.add_argument("-fr", "--filter_rate", type=float, default=0.5, help='choose filtering rate in "diverse_connect" pairing, set 1 to keep all')
parser.add_argument("-ni", "--no_index", action='store_true', help='use if do not want to use index embedding ')
parser.add_argument("-nc", "--no_clustering", action='store_true', help='use if do not want to use clustered data')
# parser.add_argument("-mm", "--model_mark", type=str, default=default_model, help='select model by mark here, acc_XXXX_ppl_XXXX')
# parser.add_argument("-mm", "--model_mark", type=str, default='acc_81.25_ppl_1.76_e13', help='select model by mark here, acc_XXXX_ppl_XXXX')
parser.add_argument("-gpu", "--gpu_id", type=int, default=0, help='input gpu id for current task, -1 to not use gpu')
parser.add_argument("-et", "--expand_target", type=str, default='train', help='select target set for expanding: "test", "train"')
parser.add_argument("-svt", "--slot_value_table", type=str, default='train', help='select use which slot value table: "fill", "train"')
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
RUN_REFILL = args.refill or args.full
with open(args.config, 'r') as con_f:
    CONFIG = json.load(con_f)


def remove_dup_pair(all_pairs):
    # =========== wait to move into source for better structure ============
    # there are still pair duplicate
    non_dup = {}
    dup_num = 0
    for p in all_pairs:
        key = p[0] + '|||' + p[1]
        if key in non_dup:
            dup_num += 1
        else:
            non_dup[key] = p
    return non_dup.values(), dup_num


def diverse_score(s, t):
    """
    calculate pairing score
    :param s: target str
    :param t: candidate str
    :return: score, edit distance, length penalty
    """
    lst_s = s.split()
    lst_t = t.split()
    length_penalty = math.exp(-abs((len(lst_s) - len(lst_t))/len(lst_s)))
    # length_penalty = math.exp(-abs((len(lst_s) - len(lst_t))/max(len(lst_s), len(lst_t))))
    e_d = sentence_edit_distance(lst_t, lst_s)
    # print(e_d * length_penalty, e_d, length_penalty, '\n', s, '\n', t)
    return e_d * length_penalty


def get_pairs_within_cluster(all_user_temp, mode="full_connect", cov_p=0.5, append_index=True):
    """
    pair utterance within a cluster
    :param all_user_temp: a list of utterance of same cluster
    :param mode: different mechanism for pairing: "full connect", "circle", "diverse_connect"
    :param cov_p: a float as percentage, used in diverse_connect, determine number of connections
    :return: paired utterance, pack in list: [[u1, p2], [u3, u4]]
    """
    ret = []
    if mode == "full_connect":
        for comb in combinations(all_user_temp, 2):
            ret.append(comb)  # use combination to avoid self to self pairs
            ret.append(comb[::-1])  # to get reverse of it

    elif mode == "circle":
        # pair each utterance with next one
        for i in range(-1, len(all_user_temp) - 1):
            ret.append([all_user_temp[i], all_user_temp[i + 1]])

    elif mode == 'diverse_connect':
        # pair each utterance with the most different x% utterance
        top_x = int(len(all_user_temp) * cov_p)
        expand_count = 0
        for temp in all_user_temp:
            top_diversity_set = sorted(all_user_temp, key=lambda x: diverse_score(temp, x), reverse=True)
            top_diversity_set = top_diversity_set[:min(top_x + 1, len(all_user_temp))]
            for ind, cand in enumerate(top_diversity_set):
                append_word = ' <%d>' % ind if append_index else ''
                ret.append([temp + append_word, cand])
            expand_count += len(top_diversity_set)
        # print('--- debug:', len(all_user_temp), expand_count)
    elif mode == 'random':
        random_target_size = 10
        for u in all_user_temp:
            for i in range(random_target_size):
                random_target = random.choice(all_user_temp)
                ret.append([u, random_target])
    return ret


def generate_data(task_name, pair_mode='diverse_connect', append_index=True, no_clustering=False, filter_rate=0.5):
    # =========== this function will move into source.prepare_data for better structure ============
    onmt_data_path = CONFIG['path']['OnmtData'] + 'SourceData/'
    raw_data_path = CONFIG['path']['ClusteringResult']
    if no_clustering:
        # all_file = list(filter(lambda x: '.json' in x and '_nc' in x, os.listdir(raw_data_path)))
        all_file = list(filter(lambda x: '.json' in x and '_nc' in x and 'atis' in x, os.listdir(raw_data_path)))
    else:
        # all_file = list(filter(lambda x: '.json' in x and '_nc' not in x, os.listdir(raw_data_path)))
        all_file = list(filter(lambda x: '.json' in x and '_nc' not in x and 'atis' in x, os.listdir(raw_data_path)))

    if not os.path.isdir(onmt_data_path):
        os.makedirs(onmt_data_path)

    pair_mode_str = '' if pair_mode == 'diverse_connect' else '_' + pair_mode
    no_index_str = '' if append_index else '_ni'
    no_filtering_str = '' if filter_rate < 1 else '_nf'
    for f in all_file:
        f_mark = f.replace(".json", '')
        print('=== Start to gen-data for: %s === ' % f_mark)
        tmp_pair_mode = pair_mode
        # tmp_pair_mode = 'circle' if 'test' in f_mark else pair_mode  # no need to have multi-source in source
        with open(raw_data_path + f, 'r') as reader, \
                open(onmt_data_path + f_mark + pair_mode_str + no_index_str + no_filtering_str + '_tgt.txt', 'w') as tgt_writer, \
                open(onmt_data_path + f_mark + pair_mode_str + no_index_str + no_filtering_str + '_src.txt', 'w') as src_writer:
            json_data = json.load(reader)
            all_pairs = []
            dup_num, null_num, bad_num = 0, 0, 0
            bad_cluster_num = 0
            for cluster_item in json_data.values():
                all_user_temp = []
                # ======== remove dup user-templates in same cluster ========
                raw_all_user_temp = [item["user_temp"] for item in cluster_item]
                non_dup_all_user_temp = set(raw_all_user_temp)

                # ======== remove clusters with no pairs =======
                if len(non_dup_all_user_temp) < 2:
                    bad_cluster_num += 1
                    continue
                dup_num += (len(raw_all_user_temp) - len(non_dup_all_user_temp))
                print("Cluster size change: remove duplicate",
                      len(raw_all_user_temp), len(non_dup_all_user_temp))

                # ======= filter temps by simple rule ======
                for user_temp in non_dup_all_user_temp:
                    # Commented, as in fact there is no reason to do such filtering
                    # # remove "thanks" included meaningless utterance and bad case
                    # if (len(user_temp.split()) <= 7 and ('thank' in user_temp or 'thanks' in user_temp)) or \
                    #         ('no response needed' in user_temp):
                    #     bad_num += 1
                    #     continue

                    # ====== fix null utterance bug =====
                    if user_temp.strip() == '':
                        null_num += 1
                        continue
                    all_user_temp.append(user_temp)

                # ========= pair utterance in current cluster, store result into all pairs ========
                all_pairs.extend(get_pairs_within_cluster(all_user_temp, mode=tmp_pair_mode, cov_p=filter_rate, append_index=append_index))

            # ======== remove duplicated pairs to avoid unbalanced data ========
            filtered_all_pairs, pair_dup_num = remove_dup_pair(all_pairs)
            print('%d dup tmp , %d null pairs, %d bad pairs, %d pair dup, %d bad cluster' % (
                dup_num, null_num, bad_num, pair_dup_num, bad_cluster_num))
            for ind, p in enumerate(filtered_all_pairs):
                tgt_writer.write(' '.join(low_case_tokenizer(p[1])) + '\n')
                src_writer.write(' '.join(low_case_tokenizer(p[0])) + '\n')
                if ind % 10000 == 0:
                    print(ind, 'pairs processed')


def refill_template(task, target_file, split_rate, slot_value_table):
    re_filling(CONFIG, task=task, target_file_name=target_file, split_rate=split_rate, slot_value_table=slot_value_table)


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
        for line in proc.stdout.readlines():
            print(line)
            writer.write(line.rstrip().decode("utf-8") + '\n')
            if b'error' in line.lower() and b'check_error' not in line.lower():
                raise RuntimeError


def main():
    if GEN_DATA:
        generate_data(
            task_name=TASK_NAME,
            pair_mode=args.pair_mode,
            append_index=(not args.no_index),
            no_clustering=args.no_clustering,
            filter_rate=args.filter_rate
        )
        # generate_full_pair_data()
    no_index_str = '_ni' if args.no_index else ''
    pair_mod_str = '' if args.pair_mode == 'diverse_connect' else '_' + args.pair_mode
    no_filtering_str = '' if args.filter_rate < 1 else '_nf'

    all_cluster_method = ['_nc'] if args.no_clustering else ['_leak-gan']
    # all_cluster_method = ['_nc'] if args.no_clustering else ['_intent-slot']
    # all_cluster_method = ['_nc'] if args.no_clustering else CONFIG['experiment']['cluster_method']

    # for split_rate in CONFIG['experiment']['train_set_split_rate']:
    for split_rate in [4478]:
        for cluster_method in all_cluster_method:
            # Customize these parameters for OpenNMT tool

            param_replace_table = {
                '<DATA_MARK>': TASK_NAME,
                '<DATA_DIR>': CONFIG['path']['OnmtData'] + 'SourceData',
                '<RESULT_DIR>': CONFIG['path']['OnmtData'] + 'Result',
                # '<MODEL_MARK>': args.model_mark,
                '<PAIR_MOD>': pair_mod_str,
                '<NO_INDEX>': no_index_str,
                '<NO_FILTERING>': no_filtering_str,
                '<GPU>': '' if args.gpu_id < 0 else ('-gpu %d' % args.gpu_id),
                '<EXPAND_TGT>': args.expand_target,
                '<SPLIT_RATE>': str(split_rate),
                '<CLUSTER_METHOD>': cluster_method,
            }
            print('Debug', param_replace_table)
            param_config = dress_param_with_config(CONFIG['onmt'], param_replace_table)
            if COOK_DATA:
                # to get word embedding and dict
                call_onmt('prepare_data' + TASK_NAME + cluster_method + str(split_rate) + pair_mod_str + no_index_str + no_filtering_str, param_config['prepare_data'])
            if RUN_TRAIN:
                call_onmt('train' + TASK_NAME + cluster_method + str(split_rate) + pair_mod_str + no_index_str + no_filtering_str, param_config['train'])
            if RUN_TEST:
                call_onmt('test' + TASK_NAME + cluster_method + str(split_rate) + pair_mod_str + no_index_str + no_filtering_str, param_config['test'])
            if RUN_REFILL:
                refill_template(TASK_NAME, TASK_NAME + cluster_method + str(split_rate) + pair_mod_str + no_index_str + no_filtering_str + '_pred.txt',  split_rate, args.slot_value_table)


if __name__ == "__main__":
    main()
    print('Notice! task option NOT affect gd, but do affect: cd, trn, tst.')
    # print("!!!!!!!!!!!!!!!!!! run in debug mode !!!!!!!!!!!!!!")
    print('Warn! Turn off NOISE_TRANLATE in onmt\'s translator.py')
