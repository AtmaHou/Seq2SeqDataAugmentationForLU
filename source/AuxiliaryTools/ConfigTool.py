# coding: utf-8
import json
import os
from collections import OrderedDict


def create_dir(p_list):
    new_folder_num = 0
    for p in p_list:
        if type(p) == dict:
            new_folder_num += create_dir(p.values())
        elif not os.path.isdir(p):
            os.makedirs(p)
            new_folder_num += 1
    return new_folder_num


def update_config(data_root="/users4/ythou/Projects/TaskOrientedDialogue/data/", config_path="../../config.json"):
    config = {
        'path': {
            "DataRoot": data_root,
            "RawData": {
                'stanford': data_root + "Stanford/",
                'stanford_labeled': data_root + "StanfordLabeled/",
                'atis': data_root + "Atis/",
            },
            "ClusteringResult": data_root + "ClusteringResult/",
            "GenerationResult": data_root + "GenerationResult/",
            "OnmtData": data_root + "OnmtData/",
            "Evaluate": data_root + "Evaluate/",
            "Embedding": data_root + 'Embedding/'
        },
        'onmt': {
            'prepare_data': ['python3', './OpenNMT/preprocess.py', '-train_src', '<DATA_DIR>/train_<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>_src.txt', '-train_tgt', '<DATA_DIR>/train_<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>_tgt.txt', '-valid_src', '<DATA_DIR>/dev_<DATA_MARK><CLUSTER_METHOD>1<PAIR_MOD><NO_INDEX><NO_FILTERING>_src.txt', '-valid_tgt', '<DATA_DIR>/dev_<DATA_MARK><CLUSTER_METHOD>1<PAIR_MOD><NO_INDEX><NO_FILTERING>_tgt.txt', '-save_data', '<RESULT_DIR>/processed_<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>'],
            'train': ['python3', './OpenNMT/train.py', '-data', '<RESULT_DIR>/processed_<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>', '-save_model', '<RESULT_DIR>/<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>-model', '<GPU>'],
            'test': ['python3', './OpenNMT/translate.py', '-model', '<RESULT_DIR>/<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>-model.pt', '-src', '<DATA_DIR>/<EXPAND_TGT>_<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>_src.txt', '-output', '<RESULT_DIR>/<DATA_MARK><CLUSTER_METHOD><SPLIT_RATE><PAIR_MOD><NO_INDEX><NO_FILTERING>_pred.txt', '-replace_unk', '-verbose', '-n_best', '10', '<GPU>']
        },
        'gen_with_label': {
            'prepare_data': ['python3', './OpenNMT/preprocess.py',
                             '-train_src', '<DATA_DIR>/train_<TRAIN_FILE_TAIL>_src.txt',
                             '-train_tgt', '<DATA_DIR>/train_<TRAIN_FILE_TAIL>_tgt.txt',
                             '-valid_src', '<DATA_DIR>/dev_<DEV_FILE_TAIL>_src.txt',
                             '-valid_tgt', '<DATA_DIR>/dev_<DEV_FILE_TAIL>_tgt.txt',
                             '-save_data', '<RESULT_DIR>/processed_<TRAIN_FILE_TAIL>'],
            'train': ['python3', './OpenNMT/train.py',
                      '-data', '<RESULT_DIR>/processed_<TRAIN_FILE_TAIL>',
                      '-save_model', '<RESULT_DIR>/<TRAIN_FILE_TAIL>-model',
                      '<GPU>'],
            'test': ['python3', './OpenNMT/translate.py',
                     '-model', '<RESULT_DIR>/<TRAIN_FILE_TAIL>-model.pt', '-src',
                     '<DATA_DIR>/train_<TRAIN_FILE_TAIL>_src.txt',
                     '-output', '<RESULT_DIR>/<TRAIN_FILE_TAIL>_pred.txt',
                     '-replace_unk', '-verbose', '-n_best', '5', '<GPU>']
        },
        'slot_filling': {
            'train_and_test': ['python3', './source/Evaluate/slot_filling_bilstm.py', '-t', '<TASK_NAME>', '-s', '<SEED>', '-sr', '<SPLIT_RATE>', '-cm', '<CLUSTER_METHOD>', '<EXTEND>', '<REFILL_ONLY>']
        },
        'experiment': {
            # 'train_set_split_rate': [515],  # for ablation test
            'train_set_split_rate': [129, 515, 4478],
            # 'train_set_split_rate': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2],
            # 'train_set_split_rate': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.5, 1],
            # 'train_set_split_rate': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1],
            'cluster_method': ['_intent-slot', '_intent', '_slot']
        }
    }
    with open(config_path, 'w') as writer:
        json.dump(config, writer, indent=2)
    new_folder_num = 0
    # create folder if not exist
    if not os.path.isdir(config['path']['DataRoot']):
        os.makedirs(config['path']['DataRoot'])
        new_folder_num += 1

    new_folder_num += create_dir(config['path'].values())
    print('config updated, make %d new folder to fit config setting' % new_folder_num)

if __name__ == "__main__":
    update_config("/users4/ythou/Projects/TaskOrientedDialogue/data/")  # For my linux server setting
    # update_config("E:/Projects/Research/TaskOrientedDialogue/data/")  # For my windows setting
