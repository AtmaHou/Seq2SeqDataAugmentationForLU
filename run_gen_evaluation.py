# coding: utf-8

import argparse
import json
from source.Evaluate.gen_eval import appearance_check
from set_config import refresh_config_file
import copy
import subprocess
from multiprocessing import Process, Queue, current_process, freeze_support, Manager
N_THREAD = 20

# ============ Args Process ==========
parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--task", type=str, default='weather_labeled', help="choose task:  navigate_labeled, schedule_labeled, weather_labeled, navigate, schedule, weather")

# parser.add_argument("--topx", type=int, default=5, help="select using topx of generated sentences")


# Deep Customize
parser.add_argument('--config', default='./config.json', help="specific a config file by path")
args = parser.parse_args()

# ============ Refresh Config ==========
refresh_config_file(args.config)

# ============ Settings ==========
with open(args.config, 'r') as con_f:
    CONFIG = json.load(con_f)

TASK_NAME_LST = ['atis_labeled']
# TASK_NAME_LST = ['navigate_labeled', 'schedule_labeled', 'weather_labeled']
CLUSTER_METHOD_LST = ['_intent-slot', '_nc', '_leak-gan']
# CLUSTER_METHOD_LST = CONFIG['experiment']['cluster_method'] + ['_nc']
# SPLIT_RATE_LST = [1]
SPLIT_RATE_LST = [4478]
# SPLIT_RATE_LST = [515]
PAIRING_MODE_LST = ['', '_full_connect', '_circle', '_random']
INDEX_CHOICE_LST = ['', '_ni', '_nf']
TOP_X = 10
USE_METRIC = [
    # "Not Appeared",
    # "Total",
    # "Unique",
    "Unique New",
    "Avg. Distance for new",
    "Avg. Distance for augmented",
    "Avg. Distance for source",
    # 'Avg. Length',

    'source_distinct_1',
    'source_distinct_2',
    'source_unigram',
    'source_bigram',
    'source_total_word',

    'augmented_distinct_1',
    'augmented_distinct_2',
    'augmented_unigram',
    'augmented_bigram',
    'augmented_total_word',

    'source_size',
    'generated_new_size',
    'augmented_size',
]


def get_file_tail(task_name, cluster_method, split_rate, pairing_mod, index_choice):
    file_tail = f"{task_name}{cluster_method}{str(split_rate)}{pairing_mod}{index_choice}"
    return file_tail


def gen_evaluation_thread(task_queue, done_queue):
    for param in iter(task_queue.get, 'STOP'):
        file_tail = get_file_tail(** param)
        ret = copy.copy(param)
        ret['eval_result'] = appearance_check(
            result_file=CONFIG['path']["OnmtData"] + "Result/" + file_tail + '_gen_eval.log',
            test_what_file=CONFIG['path']["OnmtData"] + "Result/" + file_tail + '_pred.txt',
            in_what_file=CONFIG['path']["OnmtData"] + "SourceData/train_" + file_tail + '_src.txt',
            top_x=TOP_X
        )
        done_queue.put(ret)


def format_output(result_table, output_file='./log/gen_eval_table.log'):
    output_table = []
    all_column_name = ['model_name']
    for row_name in result_table:
        temp_row = [row_name]
        for task_name in result_table[row_name]:
            for metric in USE_METRIC:
                column_name = f"{task_name}_{metric}"
                if column_name not in all_column_name:
                    all_column_name.append(column_name)
                temp_row.append('%.2f' % (result_table[row_name][task_name][metric]))
        output_table.append(temp_row)
    output_table = sorted(output_table, key=lambda x:x[0])
    with open(output_file, 'w') as writer:
        print('\t'.join(all_column_name))
        writer.write('\t'.join(all_column_name) + '\n')
        for row in output_table:
            writer.write('\t'.join(row) + '\n')
            print('\t'.join(row))


def gen_evaluation(task_name_lst, cluster_method_lst, split_rate_lst, pairing_mode_lst, index_choice_lst):
    result_table = {}

    task_queue, done_queue, task_n = Queue(), Queue(), 0
    for task_name in task_name_lst:
        for cluster_method in cluster_method_lst:
            for split_rate in split_rate_lst:
                for pairing_mod in pairing_mode_lst:
                    for index_choice in index_choice_lst:
                        param = {
                            "task_name": task_name,
                            "cluster_method": cluster_method,
                            'split_rate': split_rate,
                            'pairing_mod': pairing_mod,
                            'index_choice': index_choice,
                        }
                        task_queue.put(param)
                        task_n += 1
    print(task_n,'Tasks Building')
    for t in range(N_THREAD):
        task_queue.put('STOP')
    for t in range(N_THREAD):
        Process(target=gen_evaluation_thread, args=(task_queue, done_queue)).start()
    print("Start multi-thread Processing")
    # collect the results below
    for t in range(task_n):
        thread_return = done_queue.get()
        # print('=== thread return ===', thread_return)
        if 'no_file' in thread_return['eval_result']:
            print('--- debug ---', (thread_return['eval_result']['no_file']))
            pass
        else:
            model_name = thread_return['cluster_method'] + str(thread_return['split_rate']) + thread_return['pairing_mod'] + thread_return['index_choice']
            if model_name not in result_table:
                result_table[model_name] = {}
            # if thread_return['task_name'] not in  result_table[model_name]:
            result_table[model_name][thread_return['task_name']] = thread_return['eval_result']
        print(t + 1, 'task finished.')
    # print(result_table)
    format_output(result_table)

if __name__ == "__main__":
    gen_evaluation(
        task_name_lst=TASK_NAME_LST,
        cluster_method_lst=CLUSTER_METHOD_LST,
        split_rate_lst=SPLIT_RATE_LST,
        pairing_mode_lst=PAIRING_MODE_LST,
        index_choice_lst=INDEX_CHOICE_LST
    )
