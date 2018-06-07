# coding: utf-8
import json
import os
import re
LOG_DIR = '../../log/'
CONFIG_PATH = '../../config.json'
with open(CONFIG_PATH, 'r') as reader:
    CONFIG = json.load(reader)
RANDOM_SEED = 100
# EXTEND_SETTING = ['extend_']
EXTEND_SETTING = ['extend_', '']
RFO_SETTING = ['_refill-only', '']
TASK_SETTING = ['atis_labeled']
# TASK_SETTING = ['navigate_labeled', 'schedule_labeled', 'weather_labeled']

RES_TABLE = {}

def collect_and_format_slot_filling_result_from_log():
    all_log_file = os.listdir(LOG_DIR)
    all_column_name = []
    all_split_rate = []
    for file_name in all_log_file:
        if 'slot-filling' in file_name:
            with open(f"{LOG_DIR}{file_name}", 'r') as reader:
                split_rate = re.findall(r"[-+]?\d*\.\d+|\d+", file_name)[0]
                column_name = file_name.replace('slot-filling', '').replace('log', '').replace('_labeled', '').replace(str(split_rate), '')
                result = re.findall('slot_filling_bilstm.py : INFO  test F1-value: (.*)\n', reader.read())[0]
                if split_rate not in RES_TABLE:
                    RES_TABLE[split_rate] = {}
                    all_split_rate.append(split_rate)
                RES_TABLE[split_rate][column_name] = result
                if column_name not in all_column_name:
                    all_column_name.append(column_name)

    all_column_name = ['split_rate'] + sorted(all_column_name)
    all_split_rate = sorted(all_split_rate, key=lambda x: float(x))

    output_rows = ['\t'.join(all_column_name)]
    for sr in all_split_rate:
        temp_row  = [str(sr)]
        for c in all_column_name[1:]:
            temp_row.append(RES_TABLE[sr][c])
        output_rows.append('\t'.join(temp_row))
    with open(f"{LOG_DIR}slot_filling_result", 'w') as writer:
        for line in output_rows:
            writer.write(f'{line}\n')

def collect_and_format_slot_filling_result_from_prf():
    prf_dir = CONFIG['path']['Evaluate'] + 'SlotFilling/prf/'
    all_column_name = ['split_rate']
    # all_split_rate = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.5, 1]
    all_split_rate = CONFIG['experiment']['train_set_split_rate']
    for split_rate in all_split_rate:
    # for split_rate in :
        if split_rate not in RES_TABLE:
            RES_TABLE[split_rate] = {}
        for task in TASK_SETTING:
            for cluster_method in CONFIG['experiment']['cluster_method']:
                # et
                for et_str in EXTEND_SETTING:
                    # rfo
                    for rfo_str in RFO_SETTING:
                        if rfo_str and not et_str:
                            continue
                        prf_file_name = f'{et_str}{task}_weight2-lstm-RandomSeed-{RANDOM_SEED}{cluster_method}{rfo_str}{split_rate}_test'
                        column_name = f'{task}{cluster_method}{"-et" if et_str else ""}{"-rof" if rfo_str else ""}'
                        if column_name not in all_column_name:
                            all_column_name.append(column_name)
                        try:
                            with open(prf_dir + prf_file_name, 'r') as reader:
                                f_value = re.findall('accuracy:.*?; precision:.*?; recall:.*?; FB1:(.*?)\n', reader.read())[0].strip()
                                # print(66666666666666, prf_file_name, f_value)
                        except FileNotFoundError:
                            # print('No file:', prf_file_name)
                            f_value = 'N/A'
                        RES_TABLE[split_rate][column_name] = f_value
                        # print(split_rate, task, et_str, rfo_str)

    output_rows = ['\t'.join(all_column_name)]
    # print(all_column_name)
    for sr in all_split_rate:
        temp_row = [str(sr)]
        for c in all_column_name[1:]:
            temp_row.append(RES_TABLE[sr][c])
        output_rows.append('\t'.join(temp_row))
    with open(f"{LOG_DIR}slot_filling_result", 'w') as writer:
        for line in output_rows:
            writer.write(f'{line}\n')


def get_domain(ind):
    if ind < 9:
        return 'n'
    if 9 <= ind < 18:
        return 's'
    if 18 <= ind:
        return 'w'

def get_model(ind):

    if ind % 3 == 0:
        return 'r'
    if ind % 3 == 1:
        return 'f'
    if ind % 3 == 2:
        return 'o'

def get_average_from_all_result():
    row_name = ["navigate_intent-slot", "navigate_labeled_intent-slot-et-rof", "navigate_labeled_intent-slot-et", "navigate_intent", "navigate_labeled_intent-et-rof", "navigate_labeled_intent-et", "navigate_slot", "navigate_labeled_slot-et-rof", "navigate_labeled_slot-et", "schedule_intent-slot", "schedule_labeled_intent-slot-et-rof", "schedule_labeled_intent-slot-et", "schedule_intent", "schedule_labeled_intent-et-rof", "schedule_labeled_intent-et", "schedule_slot", "schedule_labeled_slot-et-rof", "schedule_labeled_slot-et", "weather_intent-slot", "weather_labeled_intent-slot-et-rof", "weather_labeled_intent-slot-et", "weather_intent", "weather_labeled_intent-et-rof", "weather_labeled_intent-et", "weather_slot", "weather_labeled_slot-et-rof", "weather_labeled_slot-et"]
    # paste results here
    # rfo vs rf + gen | train table only
    all_result = []
    input_file = '/users4/ythou/Projects/TaskOrientedDialogue/code/DialogueDiversification/log/slot_filling_result'
    result_file = '/users4/ythou/Projects/TaskOrientedDialogue/code/DialogueDiversification/log/merged_slot_filling_result'
    with open(input_file, 'r') as reader:
        all_lines = reader.readlines()
    for line in all_lines[1:]:
        one_line_res = []
        for score in line.split('\t')[1:]:
            one_line_res.append(float(score))
        all_result.append(one_line_res)

    merged_res = {
        'n':{'o':[], 'r':[], 'f':[]},
        's':{'o':[], 'r':[], 'f':[]},
        'w':{'o':[], 'r':[], 'f':[]}
    }
    final_res = {
        'n': {'o': [], 'r': [], 'f': []},
        's': {'o': [], 'r': [], 'f': []},
        'w': {'o': [], 'r': [], 'f': []}
    }
    for i, row in enumerate(all_result):
        temp_res = {
            'n': {'o': [], 'r': [], 'f': []},
            's': {'o': [], 'r': [], 'f': []},
            'w': {'o': [], 'r': [], 'f': []}
        }
        for j, c in enumerate(row):
            model = get_model(j)
            domain = get_domain(j)
            temp_res[domain][model].append(c)
            print('debug', row_name[j], domain, model, '\n')
        for d in temp_res:
            for m in temp_res[d]:
                merged_res[d][m].append(temp_res[d][m])
    with open(result_file, 'w') as writer:
        line_str = ''
        for d in merged_res:
            for m in merged_res[d]:
                line_str += '%s-%s\t' % (d, m)
        print(line_str)
        writer.write(line_str + '\n')
        for r_n in range(len(all_result)):
            line_str = ''
            for d in merged_res:
                for m in merged_res[d]:
                    if m == 'f':
                        line_str += '%.2f\t' % (max(merged_res[d][m][r_n]))
                        # line_str += '%.2f\t' % (sum(merged_res[d][m][r_n]) / 3)
                    else:
                        line_str += '%.2f\t' % (sum(merged_res[d][m][r_n]) / 3)
            print(line_str)
            writer.write(line_str + '\n')
    print("notice ori-order")

if __name__ == '__main__':
    # collect_and_format_slot_filling_result_from_log()  # abandoned
    collect_and_format_slot_filling_result_from_prf()
    print('Notice :change setting for task and random et.al. before running if collect from prf')
    print('Evaluating task', TASK_SETTING)
    # get_average_from_all_result()
