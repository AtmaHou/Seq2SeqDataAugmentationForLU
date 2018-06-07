# coding:utf-8
import re
import json
from source.AuxiliaryTools.nlp_tool import sentence_edit_distance
import random
from multiprocessing import Process, Queue, current_process, freeze_support, Manager
import copy
N_THREAD = 20
TASK_SIZE = 500
CONTEXT_WINDOW_SIZE = 2
FULL_SLOT_TABLE_AVAILABLE = True
CONTEXT_REFILL_RATE = 0.5  # Refill slot by context match by this rate
DEBUG = False


def old_candidate_num(d):
    ret = 0
    for value in d.values():
        ret += len(value)
    return ret


def candidate_num(d):
    return len(d)


def old_re_fill_sentences(lines, temp_query_dict, full_query_dict):
    """
    Step1: exact match context(temp) in temp_query_dict, if matched, refilling randomly and go to step3
    Step2: Find the most similar context(temp), and refilling
    Step3: Refill the remained slot with full_query_dict
    :param lines: list of tuple (template sentence, line_id)
    :param temp_query_dict: A dict for query, key is temp and value is possible slot candidate
    :param full_query_dict: A dic for query, contains all slot name - possible slot value pair
    :return: sentence after surface realization
    """
    res_lst = []
    r1, r2, r3 = 0, 0, 0
    for line, line_id in lines:
        tmp_res = line
        # Step1
        if line in temp_query_dict:
            for slot_name in temp_query_dict[line]:
                select_slot_value = random.choice(temp_query_dict[line][slot_name])
                tmp_res = tmp_res.replace('<' + slot_name + '>', select_slot_value)
                r1 += 1
        # Step2
        else:
            target_temp_word_lst = line.split()
            most_similar_temp = ""
            min_distance = len(target_temp_word_lst)
            for temp in temp_query_dict:
                current_distance = sentence_edit_distance(temp, target_temp_word_lst)
                if current_distance < min_distance:
                    min_distance = current_distance
                    most_similar_temp = temp
                elif current_distance == min_distance and most_similar_temp \
                        and old_candidate_num(temp_query_dict[temp]) > old_candidate_num(temp_query_dict[most_similar_temp]):
                    min_distance = current_distance
                    most_similar_temp = temp
            # print('Debug', min_distance, most_similar_temp, "|||", target_temp_word_lst)
            if most_similar_temp:  # Fill with the slots within most similar temp
                for slot_name in temp_query_dict[most_similar_temp]:
                    select_slot_value = random.choice(temp_query_dict[most_similar_temp][slot_name])
                    tmp_res = tmp_res.replace('<' + slot_name + '>', select_slot_value)
                    r2 += 1
        # Step3
        slot_name_lst = re.findall("<(.*?)>", tmp_res)
        if slot_name_lst:
            for slot_name in slot_name_lst:
                select_slot_value = random.choice(full_query_dict[slot_name])
                tmp_res = tmp_res.replace("<" + slot_name + ">", select_slot_value)
                r3 += 1
        # recheck
        if re.findall("<.*?>", tmp_res):
            print("Error; unfinished re-filling!", tmp_res, '2333333333')
        res_lst.append((tmp_res, line_id))
    return res_lst, r1, r2, r3


def extract_slot_context(word_lst):
    """
    retuen a slot list,
    :param word_lst:
    :return:
    """
    slot_lst = []
    context_lst = []
    for ind, word in enumerate(word_lst):
        if '<' in word and '>':
            slot_lst.append(word)
            context_text_word_lst = word_lst[max(ind - CONTEXT_WINDOW_SIZE, 0): ind + CONTEXT_WINDOW_SIZE + 1]
            context_lst.append(context_text_word_lst)
        else:
            slot_lst.append(None)
            context_lst.append(None)
    return slot_lst, context_lst


def expand_tmp_res_and_get_label_lst(tmp_res, slot_lst):
    final_res = []
    label_lst = []
    for tmp_res_word, slot_word in zip(tmp_res, slot_lst):
        if type(tmp_res_word) == str:
            final_res.append(tmp_res_word)
            label_lst.append('O')
        elif type(tmp_res_word) == list:
            slot_name = slot_word.replace('<', '').replace('>', '')
            final_res.extend(tmp_res_word)
            label_lst.append('B-' + slot_name)
            for i_value in tmp_res_word[1:]:
                label_lst.append('I-' + slot_name)
        else:
            raise TypeError
    return final_res, label_lst


def re_fill_sentences(lines, context_query_dict, full_query_dict, refill_only, full_slot_table=FULL_SLOT_TABLE_AVAILABLE):
    """
    Step1: exact match context(temp) in temp_query_dict, if matched, refilling randomly and go to step3
    Step2: For each slot name find the most similar context(temp), and refilling
    Step3: Refill the remained slot with full_query_dict
    :param lines: list of tuple (template sentence, line_id)
    :param context_query_dict: A dict for query, key is temp and value is possible slot candidate
    :param full_query_dict: A dic for query, contains all slot name - possible slot value pair
    :return: sentence after surface realization
    """
    res_lst = []
    r1, r2, r3 = 0, 0, 0
    for line, line_id in lines:
        debug_lst = []
        if refill_only:
            line = re.sub('<\d+>', '', line)
        word_lst = line.split()
        tmp_res = word_lst
        slot_lst, context_lst = extract_slot_context(word_lst)
        for ind in range(len(slot_lst)):
            slot_word = slot_lst[ind]
            if slot_word and slot_word != '<unk>':
                context_text_word_lst = context_lst[ind]

                slot_name = slot_word.replace('<', '').replace('>', '')
                context_text = ' '.join(context_text_word_lst)

                # judge weather is slot  & randomly using other slot & judge weather have these slot name
                if slot_name and random.random() <= CONTEXT_REFILL_RATE and slot_name in context_query_dict:
                    # Step1
                    if context_text in context_query_dict[slot_name]:
                            select_slot_value = random.choice(context_query_dict[slot_name][context_text])
                            # select_slot_value is str
                            tmp_res[ind] = select_slot_value.split()
                            debug_lst.append([1, slot_word, select_slot_value, tmp_res])
                            r1 += 1
                    # Step2
                    else:
                        most_similar_temp = ""
                        min_distance = len(context_text_word_lst)
                        for candidate_context in context_query_dict[slot_name]:
                            current_distance = sentence_edit_distance(candidate_context, context_text_word_lst)
                            if current_distance < min_distance:
                                min_distance = current_distance
                                most_similar_temp = candidate_context
                            # select the candidate with more possible values
                            elif current_distance == min_distance and most_similar_temp \
                                    and candidate_num(context_query_dict[slot_name][candidate_context]) >  \
                                    candidate_num(context_query_dict[slot_name][most_similar_temp]):
                                min_distance = current_distance
                                most_similar_temp = candidate_context
                        if most_similar_temp:  # Fill with the slots within most similar temp
                            select_slot_value = random.choice(context_query_dict[slot_name][most_similar_temp])
                            # select_slot_value is str
                            tmp_res[ind] = select_slot_value.split()
                            debug_lst.append([2, slot_word, select_slot_value, tmp_res])
                            r2 += 1
        # Step3
        tmp_res_copy = copy.deepcopy(tmp_res)
        for ind, slot_word in enumerate(tmp_res_copy):
            if '<' in slot_word and '>' in slot_word and type(slot_word) == str and slot_word != '<unk>':
                slot_name = slot_word.replace('<', '').replace('>', '')

                try:
                    select_slot_value = random.choice(full_query_dict[slot_name])
                    tmp_res[ind] = select_slot_value.split()
                    debug_lst.append([3, "<" + slot_name + ">", select_slot_value, tmp_res])
                    r3 += 1
                except KeyError:
                    print('================Key Warning \nslot_lst:', slot_word, '\nline:', line, '\ntmp_res:', tmp_res, '\nslot_lst', slot_lst, '\ncontext_lst', context_lst)
        # print('================\nslot_name_lst:', slot_name_lst, '\nline:', line, '\ntmp_res:', tmp_res, '\nslot_lst',
        #       slot_lst, '\ncontext_lst', context_lst)

        # recheck
        tmp_res_copy = copy.deepcopy(tmp_res)
        for ind, x in enumerate(tmp_res_copy):
            if '<' in x and '>' in x and x != '<unk>' and type(x) == str:
                try:  # to capture the weird <1> appear problem
                    int(x.replace('<', '').replace('>', ''))
                    tmp_res[ind] = '<unk>'
                except ValueError:
                    print("Error; unfinished re-filling!", tmp_res, '!!!!!!!!!!!!!!!!!!!!!!')
        # missed_set = list(filter(lambda x: '<' in x and '>' in x and x != '<unk>' and type(x) == str, tmp_res))
        # if missed_set:
        #     print("Error; unfinished re-filling!", missed_set, tmp_res, '!!!!!!!!!!!!!!!!!!!!!!')

        if DEBUG:
            for db in debug_lst:
                print(db)
        final_res, label_lst = expand_tmp_res_and_get_label_lst(tmp_res, slot_lst)
        res_lst.append((final_res, line_id, label_lst))
    return res_lst, r1, r2, r3


def re_filling_thread(task_queue, done_queue):
    for param in iter(task_queue.get, 'STOP'):
        ret = re_fill_sentences(** param)
        done_queue.put(ret)


def re_filling(config, task, target_file_name='navigate1_pred.txt', split_rate=1, slot_value_table='train', refill_only=False):
    # print('?????', task, target_file_name, split_rate, slot_value_table, refill_only)
    result_dir = config['path']['OnmtData'] + 'Result/'
    if not refill_only:
        input_dir = config['path']['OnmtData'] + 'Result/'
    else:
        input_dir = config['path']['OnmtData'] + 'SourceData/'
    target_file_path = input_dir + target_file_name
    result_file_path = result_dir + target_file_name.replace('.txt', '_refilled.txt')
    for_conll_file_path = result_dir + target_file_name.replace('.txt', '_for-conll.json')
    dict_dir = config['path']['ClusteringResult']

    temp_query_path = dict_dir + task + "_temp-query.dict"
    context_query_path = dict_dir + task + str(split_rate) + "_context-query.dict"
    if slot_value_table == 'full':
        full_query_path = dict_dir + task + "_full-query.dict"
        # print('=!!!!!!!!!!!!! full_query_path', full_query_path)
    elif slot_value_table == 'train':
        full_query_path = dict_dir + task + str(split_rate) + "_train_full-query.dict"
    else:
        print('Error: Wrong setting for slot value table, only train and full are supported')
        raise RuntimeError

    all_results = []
    # re-filling case statistic
    all_r1, all_r2, all_r3 = 0, 0, 0
    with open(temp_query_path, 'r') as temp_query_file, \
            open(full_query_path) as full_query_file, \
            open(context_query_path) as context_query_file:
        temp_query_dict = json.load(temp_query_file)
        full_query_dict = json.load(full_query_file)
        context_query_dict = json.load(context_query_file)
    print("stat re-filling for %s" % target_file_path)

    # debug_line = 'give me directions to my <poi_type> food the <traffic_info> .'
    # print(re_fill_sentence(debug_line.replace('\n', ''), temp_query_dict, full_query_dict))
    task_queue, done_queue, task_n = Queue(), Queue(), 0
    with open(target_file_path, 'r') as reader, open(result_file_path, 'w') as writer, \
            open(for_conll_file_path, 'w') as for_conll_file:
        line_count = 0
        lines = []
        all_sent_set = set()
        for line in reader:
            # if '_ni' in target_file_name and line in all_sent_set:
            if ('_ni' in target_file_name or '_nf' in target_file_name) and line in all_sent_set:
                # remove duplicate for specific task
                continue
            else:
                all_sent_set.add(line)
            lines.append((line.replace('\n', ''), line_count))
            if line_count % TASK_SIZE == 0:
                param = {
                    'lines': lines,
                    'context_query_dict': context_query_dict,
                    'full_query_dict': full_query_dict,
                    'refill_only': refill_only
                }
                # param = {
                #     'lines': lines,
                #     'temp_query_dict': temp_query_dict,
                #     'full_query_dict': full_query_dict
                # }
                task_queue.put(param)
                task_n += 1
                lines = []
            line_count += 1
        # to collect the left data
        param = {
            'lines': lines,
            'context_query_dict': context_query_dict,
            'full_query_dict': full_query_dict,
            'refill_only': refill_only
        }
        # param = {
        #     'lines': lines,
        #     'temp_query_dict': temp_query_dict,
        #     'full_query_dict': full_query_dict
        # }
        task_queue.put(param)
        task_n += 1
        print("Start multi-thread Processing")
        for t in range(N_THREAD):
            task_queue.put('STOP')
        for t in range(N_THREAD):
            Process(target=re_filling_thread, args=(task_queue, done_queue)).start()
        print("All threads created")
        # collect the results below
        for t in range(task_n):
            refilled_res_lst, r1, r2, r3 = done_queue.get()
            # calculate result to utilize python's parallel loading feature
            all_results.extend(refilled_res_lst)
            all_r1 += r1
            all_r2 += r2
            all_r3 += r3
            if t * TASK_SIZE % 10000 == 0:
                print(t * TASK_SIZE, 'lines processed')
        print('Filling finished, three re-filling case statistic as follow:')
        print('find ori:%d, most same:%d, leaked slots%d' % (all_r1, all_r2, all_r3))
        sorted_result = sorted(all_results, key=lambda x: x[1])
        for res in sorted_result:
            writer.write(' '.join(res[0]) + '\n')
        json_for_conll = []
        for res in sorted_result:
            json_for_conll.append(
                {
                    'word_lst': res[0],
                    'label_lst': res[2]
                }
            )
        json.dump(json_for_conll, for_conll_file)
