# coding: utf-8

DATA_FORMAT =  'conll'
INPUT_FILE = '/users4/ythou/Projects/TaskOrientedDialogue/data/AtisRaw/atis.train'
OUTPUT_DIR = '/users4/ythou/Projects/TaskOrientedDialogue/data/Atis/'
DATA_MARK = 'atis'
TRAIN_DEV_RATE = [0.8, 0.2]
TRAIN_DEV_COUNT = [None, 500]
USE_RATE = False

def split():
    with open(INPUT_FILE, 'r') as input_f, \
            open(f'{OUTPUT_DIR}{DATA_MARK}_train', 'w') as train_f, \
            open(f'{OUTPUT_DIR}{DATA_MARK}_dev', 'w') as dev_f:
        if DATA_FORMAT == 'conll':
            all_data = input_f.read().strip().split('\n\n')
            if USE_RATE:
                train_end_ind = int(len(all_data) * TRAIN_DEV_RATE[0])
            else:
                train_end_ind = len(all_data) - TRAIN_DEV_COUNT[1]
            train_data = all_data[: train_end_ind]
            dev_data = all_data[train_end_ind:]
            print(f'train{train_end_ind}, dev{len(all_data)- train_end_ind}')
            train_f.write('\n\n'.join(train_data))
            dev_f.write('\n\n'.join(dev_data))

if __name__  == '__main__':
    split()
