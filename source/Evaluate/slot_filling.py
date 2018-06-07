# coding:utf-8
"""
Tips:
    slot label format:
    B-slot_name, there is no < or > in slot_name
"""
import json
import os
from source.AuxiliaryTools.nlp_tool import low_case_tokenizer
import re

# VERBOSE = True
VERBOSE = False
# DEBUG = False
DEBUG = True
SENT_COUNT_SPLIT = True

# def out_of_slot(i, j, tmp_word_lst, ori_word_lst):
#     """
#     to check if pointer of  ori is out of the slot area
#     :param i: index of the current slot
#     :param j:
#     :param tmp_word_lst:
#     :param ori_word_lst:
#     :return:
#     """
#     # two list keep same until next slot
#     try:
#         next_slot_id_inc = [('<' in x and '>' in x) for x in tmp_word_lst[i + 1:]].index(True)
#         inner_remained_tmp = tmp_word_lst[i + 1: i + 1 + next_slot_id_inc]
#         inner_remained_ori = ori_word_lst[j:j + next_slot_id_inc]
#         return inner_remained_ori == inner_remained_tmp
#     except:
#         # There is no next slot
#         remained_tmp = tmp_word_lst[i + 1:]  # if i + 1 exceed length, they must both be []
#         remained_ori = ori_word_lst[j:]
#         return remained_tmp == remained_ori


# def get_label_from_temp(user_temp, user_say, verbose=VERBOSE):
#     label_lst = []
#     tmp_word_lst = low_case_tokenizer(user_temp)
#     ori_word_lst = low_case_tokenizer(user_say)
#     i = 0
#     j = 0
#     while i < len(tmp_word_lst) and j < len(ori_word_lst):
#         tmp_word = tmp_word_lst[i]
#         slot_name = re.findall("<(.*?)>", tmp_word)
#         if slot_name:
#             # find a slot, align the slot word
#             slot_name = slot_name[0]
#             label_lst.append('B-' + slot_name)
#             j += 1
#             while not out_of_slot(i, j, tmp_word_lst, ori_word_lst) and j <= len(ori_word_lst):
#                 # print('i, j, tmp_word_lst, ori_word_lst', i, j, tmp_word_lst, ori_word_lst, out_of_slot(i, j, tmp_word_lst, ori_word_lst))
#                 label_lst.append('I-' + slot_name)
#                 j += 1
#             j -= 1
#         else:
#             if DEBUG and verbose:
#                 print('debug', 'i', i, 'j', j, '\n', ori_word_lst, '\n', tmp_word_lst, '\n', label_lst)
#             if ori_word_lst[j] == tmp_word_lst[i]:
#                 label_lst.append('O')
#             else:
#                 print(ori_word_lst[j], "vs", tmp_word_lst[i])
#                 print()
#                 print("ERROR fail to match non-slot word!!!!!!!!!!!")
#                 raise RuntimeError
#         j += 1
#         i += 1
#     if verbose:
#         print('ori:', ori_word_lst)
#         print('tmp:', tmp_word_lst)
#         print('label:', label_lst)
#         print('zip res:', list(zip(ori_word_lst, label_lst)))
#         print('\n')
#     if len(label_lst) != len(ori_word_lst):
#         print("Error: not equal length between label and word list!")
#         print(len(ori_word_lst), ori_word_lst)
#         print(len(tmp_word_lst), tmp_word_lst)
#         print(len(label_lst), label_lst)
#         raise RuntimeError
#     else:
#         if label_lst and ori_word_lst:
#             # Remove I-slot_name word in label lst and ori_word_lst, to equal the length with temp_word_lst
#             compressed_label_lst, compressed_ori_word_lst = zip(*filter(
#                 lambda x: 'I-' not in x[1], zip(ori_word_lst, label_lst)
#             ))
#         else:
#             compressed_label_lst, compressed_ori_word_lst = [], []
#         for l, tw, cow in zip(compressed_label_lst, tmp_word_lst, compressed_label_lst):
#             if l == 'O' and tw != cow:
#                 print("Error: label not aligned!")
#                 print(ori_word_lst)
#                 print(tmp_word_lst)
#                 print(label_lst)
#                 raise RuntimeError
#     # Result check: to find not aligned error
#     return label_lst, ori_word_lst


def get_slot_filling_data_from_cluster(cluster_data_file):
    ret = []
    all_user_say = set()
    with open(cluster_data_file, 'r') as cluster_file:
        cluster_data = json.load(cluster_file)
        for turn_lst in cluster_data.values():
            for turn in turn_lst:
                user_say = turn['user_say']
                user_temp = turn['user_temp']
                if not user_temp:
                    continue  # remove empty turn
                all_user_say.add(user_say)
                # label_lst, word_sequence = get_label_from_temp(user_temp, user_say)
                data_item = {
                    "utters": {
                        "tags": turn['label_lst'],
                        "ws": turn['user_word_lst'],
                    }
                }
                ret.append(data_item)
                # except RuntimeError:
                #     if DEBUG:
                #         print('==== debug ====', cluster_data_file, user_temp, user_say)
    return ret, all_user_say


def get_slot_filling_data_from_generation(for_conll_file_path, ori_say_set, use_topx, refilled_only=False):
# def get_slot_filling_data_from_generation(gen_f_path, refill_f_path, ori_say_set, use_topx, refilled_only=False):
    ret = []
    # with open(gen_f_path, 'r') as gen_f, open(refill_f_path, 'r') as refill_f:
    #     all_user_temp = gen_f.readlines()
    #     all_user_say = refill_f.readlines()
    #     temp_say_pairs = set(zip(all_user_temp, all_user_say))  # remove dup
    with open(for_conll_file_path) as for_conll_file:
        json_for_conll = json.load(for_conll_file)
        for ind, pair in enumerate(json_for_conll):
            if ind % 10 >= use_topx:
                continue
            elif ' '.join(pair['word_lst']) in ori_say_set:  # remove occurred refilled utterance
                continue
            try:
                # user_say = pair[1]
                # user_temp = re.sub('<\d+>', '', pair[0]) if refilled_only else pair[0]  # remove id label such as <1>
                # label_lst, word_sequence = get_label_from_temp(user_temp, user_say)
                data_item = {
                    "utters": {
                        "tags": pair['label_lst'],
                        "ws": pair['word_lst']
                    }
                }
                if ind % 10000 == 0:
                    print(ind, 'pairs finished.')
                ret.append(data_item)
            except RuntimeError:
                if DEBUG:
                    print('==== debug ====', for_conll_file_path)
                    # print('==== debug ====', gen_f_path)
        return ret


def prepare_data_to_dukehan(config, task_name='navigate_labeled', split_rate=1, use_topx=10):

    print('Processing data for: ', task_name, split_rate)
    # define dir path
    gen_result_dir = config['path']['OnmtData'] + 'Result/'
    cluster_result_dir = config['path']['ClusteringResult']
    output_data_dir = config['path']['Evaluate'] + 'SlotFilling/'
    if not os.path.isdir(output_data_dir):
        os.makedirs(output_data_dir)

    # define input file path
    refilled_data_path = gen_result_dir + task_name + str(split_rate) + '_pred_refilled.txt'
    gen_data_path = gen_result_dir + task_name + str(split_rate) + '_pred.txt'
    train_cluster_result_path = cluster_result_dir + 'train_%s%s.json' % (task_name, str(split_rate))
    test_cluster_result_path = cluster_result_dir + 'test_%s1.0.json' % task_name
    dev_cluster_result_path = cluster_result_dir + 'dev_%s1.0.json' % task_name
    all_slot_value_dict = cluster_result_dir + '%s_full-query.dict' % task_name

    # define output file path
    train_path = output_data_dir + 'train' + str(split_rate) + '.json'
    dev_path = output_data_dir + 'dev.json'
    test_path = output_data_dir + 'test.json'
    extend_train_path = output_data_dir + 'extend_train' + str(split_rate) + '.json'

    # get data label pair from cluster result
    result_for_train, all_train_user_say = get_slot_filling_data_from_cluster(train_cluster_result_path)
    result_for_test, _ = get_slot_filling_data_from_cluster(test_cluster_result_path)
    result_for_dev, _ = get_slot_filling_data_from_cluster(dev_cluster_result_path)
    # print('debug: all user', len(all_train_user_say))
    # get extra data from generation
    result_for_extend_train = get_slot_filling_data_from_generation(gen_data_path, refilled_data_path, all_train_user_say, use_topx=use_topx)

    # get all slot set
    all_slot_label = ['O']
    with open(all_slot_value_dict, 'r') as reader:
        all_slot_set = json.load(reader).keys()
    for slot_name in all_slot_set:
        all_slot_label.append('B-' + slot_name)
        all_slot_label.append('I-' + slot_name)

    print('debug', len(result_for_extend_train))
    # output to file
    with open(train_path, 'w') as train_res_f, \
            open(dev_path, 'w') as dev_res_f, \
            open(test_path, 'w') as test_res_f, \
            open(extend_train_path, 'w') as extend_train_res_f:
        train_res = {
            'tags': all_slot_label,
            'data': result_for_train
        }
        dev_res = {
            'tags': all_slot_label,
            'data': result_for_dev
        }
        test_res = {
            'tags': all_slot_label,
            'data': result_for_test
        }
        extend_train_res = {
            'tags': all_slot_label,
            'data': result_for_extend_train
        }
        json.dump(train_res, train_res_f)
        json.dump(dev_res, dev_res_f)
        json.dump(test_res, test_res_f)
        json.dump(extend_train_res, extend_train_res_f)


def format_and_output_conll_data(file_path, results):
    print('Out put to', file_path)
    with open(file_path, 'w') as writer:
        for data_item in results:
            tag_lst = data_item['utters']['tags']
            ws_lst = data_item['utters']['ws']
            for tag, ws in zip(tag_lst, ws_lst):
                writer.write('%s\t%s\n' % (ws, tag))
            writer.write('\n')


def remove_result_duplication(results):
    deduplicated_result = []
    appeared_user_say = set()
    for data_item in results:
        user_utterance = ' '.join(data_item['utters']['ws'])
        if user_utterance not in appeared_user_say:
            appeared_user_say.add(user_utterance)
            deduplicated_result.append(data_item)
    return deduplicated_result


def prepare_data_to_conll_format(config, task_name='navigate', split_rate=1, cluster_method='_intent', use_topx=10, refilled_only=False, pair_mod='', no_index='', no_filter_str='' ):

    print('Processing data for: ', task_name + cluster_method, split_rate)
    # define dir path
    gen_result_dir = config['path']['OnmtData'] + 'Result/'
    gen_source_dir = config['path']['OnmtData'] + 'SourceData/'
    cluster_result_dir = config['path']['ClusteringResult']
    output_data_dir = config['path']['Evaluate'] + 'SlotFilling/Source/'
    if not os.path.isdir(output_data_dir):
        os.makedirs(output_data_dir)

    # define input file path
    # refilled_data_path = gen_result_dir + task_name + cluster_method + str(split_rate) + '_pred_refilled.txt'
    # gen_data_path = gen_result_dir + task_name + cluster_method + str(split_rate) + '_pred.txt'
    gen_for_conll_file_path = gen_result_dir + task_name + cluster_method + str(split_rate) + pair_mod + no_index + no_filter_str +'_pred_for-conll.json'
    # rfo_refilled_data_path = gen_result_dir + 'train_' + task_name + cluster_method + str(split_rate) + '_src_refilled.txt'
    # rfo_gen_data_path = gen_source_dir + 'train_' + task_name + cluster_method + str(split_rate) + '_src.txt'
    rfo_for_conll_file_path = gen_result_dir + 'train_' + task_name + cluster_method + str(split_rate) + '_src_for-conll.json'

    train_cluster_result_path = cluster_result_dir + 'train_%s%s%s.json' % (task_name, cluster_method, str(split_rate))
    test_cluster_result_path = cluster_result_dir + 'test_%s%s1.json' % (task_name, cluster_method)
    dev_cluster_result_path = cluster_result_dir + 'dev_%s%s1.json' % (task_name, cluster_method)
    # all_slot_value_dict = cluster_result_dir + '%s_full-query.dict' % task_name

    # define output file path
    if not refilled_only:
        train_path = output_data_dir + 'train_' + task_name + cluster_method + str(split_rate) + '.conll'
        extend_train_path = output_data_dir + 'extend_train_' + task_name + cluster_method + str(split_rate) + pair_mod + no_index + no_filter_str + '.conll'
        dev_path = output_data_dir + 'dev_' + task_name + pair_mod + no_index + no_filter_str + '.conll'
        test_path = output_data_dir + 'test_' + task_name + pair_mod + no_index + no_filter_str + '.conll'
        full_corpus_path = output_data_dir + 'full_corpus_' + task_name + pair_mod + no_index + no_filter_str + '.conll'
    else:
        train_path = output_data_dir + 'train_' + task_name + cluster_method + '_refill-only' + str(split_rate) + '.conll'
        extend_train_path = output_data_dir + 'extend_train_' + task_name + cluster_method + '_refill-only' + str(split_rate) + '.conll'
        dev_path = output_data_dir + 'dev_' + task_name + '_refill-only' + '.conll'
        test_path = output_data_dir + 'test_' + task_name + '_refill-only' + '.conll'
        full_corpus_path = output_data_dir + 'full_corpus_' + task_name + '_refill-only' + '.conll'

    # get data label pair from cluster result
    result_for_train, all_train_user_say = get_slot_filling_data_from_cluster(train_cluster_result_path)
    result_for_test, _ = get_slot_filling_data_from_cluster(test_cluster_result_path)
    result_for_dev, _ = get_slot_filling_data_from_cluster(dev_cluster_result_path)

    # get extra data from generation (not include ori-train data)
    result_for_gen_extra_train = get_slot_filling_data_from_generation(gen_for_conll_file_path, all_train_user_say, use_topx=use_topx, refilled_only=False)
    # get extra data from source refilled (not include ori-train data), there is no beam search in rfo, use top 10
    result_for_rof_extra_train = get_slot_filling_data_from_generation(rfo_for_conll_file_path, all_train_user_say, use_topx=10, refilled_only=True)

    # merge to get extend train data
    if not refilled_only:
        # result_for_extend_train = result_for_train + result_for_gen_extra_train
        result_for_extend_train = result_for_train + result_for_rof_extra_train + result_for_gen_extra_train
    else:
        result_for_extend_train = result_for_train + result_for_rof_extra_train

    result_for_extend_train = remove_result_duplication(result_for_extend_train)
    # output to file
    format_and_output_conll_data(train_path, result_for_train)
    format_and_output_conll_data(dev_path, result_for_dev)
    format_and_output_conll_data(test_path, result_for_test)
    format_and_output_conll_data(extend_train_path, result_for_extend_train)
    # print('debug', len(result_for_extend_train))
    # get and output full corpus data
    full_data_mark = 4478 if SENT_COUNT_SPLIT else 1
    if split_rate == full_data_mark:
        print("Processing data for: full corpus")
        result_for_full_corpus = result_for_train + result_for_dev + result_for_test
        format_and_output_conll_data(full_corpus_path, result_for_full_corpus)
