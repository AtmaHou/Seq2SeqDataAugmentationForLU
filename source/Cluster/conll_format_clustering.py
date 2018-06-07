# coding:utf-8
"""
main code for clustering
define clustering class
and running this code to get clustering for stanford data
"""
import json
import os
from source.AuxiliaryTools.nlp_tool import  low_case_tokenizer

CONTEXT_WINDOW_SIZE = 2  # 2 is used because it is empirical feature setting in slot filling task
SENT_COUNT_SPLIT = False


def debug_all_data_print(d):
    for dm in d:
        print('========== domain: %s ===========' % dm)
        print(d[dm][:10])


class Cluster:
    def __init__(self, input_dir, result_dir, all_domain, cluster_mode='all', special_mark=''):
        self.input_dir = input_dir
        self.result_dir = result_dir
        self.all_domain = all_domain
        self.cluster_mode = cluster_mode
        self.special_mark = special_mark

        # Tips:
        # you can't just assign {} as values when build dict with dict.fromkeys()
        # because {}' one 1 is viewed as 1 address, result in that all {} in fact point to same memory
        # "for" is only secure unless you use map:
        # all_data = dict(zip(all_task, map(lambda x:{},[None] * len(all_task))))
        self.all_data_item = dict.fromkeys(all_domain)  # store all data item

        # ========= Prepare dict for temp refill ===============
        # 1. Template query dictionary
        # Saved in the cluster dir, keys of it are template texts,and the values of it are sub-dictionary
        # Sub-dict's key is slot names appeared in template, the value is list of appeared slot values.
        # 2. full query dictionary
        # Saved in cluster directory, the key is slot name, the value is list.
        # And the list contents are all possible slot values
        # 3. train full query dictionary
        # A list of dicts for different split rate se same to full query dictionary, but it only count train set appearance
        # 4. all context dictionary
        # Evolution of template query dict, record all slot's context format as below:
        # context_dict = {
        #     slot-name1: {
        #         3_word_context_text1 :set(value1, value2, value3)
        #     }
        # }
        # implemented below:

        self.all_temp_dict = dict.fromkeys(all_domain)
        self.all_full_dict = dict.fromkeys(all_domain)
        self.train_full_dict = dict.fromkeys(all_domain)
        self.all_context_dict = dict.fromkeys(all_domain)
        for key in all_domain:
            # init store all data item
            self.all_data_item[key] = []
            # init template refill dict
            self.all_temp_dict[key] = {}
            # init full query dict
            self.all_full_dict[key] = {}
            # init full query dict
            self.train_full_dict[key] = {}
            # init full query dict
            self.all_context_dict[key] = {}

    def update_dict(self, domain_name, data_item, data_label, split_rate):
        user_temp = data_item['user_temp']
        slot_name_lst = data_item['slot_name_lst']
        slot_value_lst = data_item['slot_value_lst']
        context_lst = data_item['context_lst']

        # ===== init the all_context_dict for current split_rate =====
        if split_rate not in self.all_context_dict[domain_name]:
            self.all_context_dict[domain_name][split_rate] = {}

        # ===== init the train_full_dict for current split_rate =====
        if split_rate not in self.train_full_dict[domain_name]:
            self.train_full_dict[domain_name][split_rate] = {}

        # ==== start to update =======
        for slot_name, slot_value, context in zip(slot_name_lst, slot_value_lst, context_lst):
            slot_value_str = ' '.join(slot_value)
            context_str = ' '.join(context)
            if slot_name:
                if data_label == 'train':
                    # ========== update context_dict =========
                    if slot_name not in self.all_context_dict[domain_name][split_rate]:
                        self.all_context_dict[domain_name][split_rate][slot_name] = {}
                    if context_str in self.all_context_dict[domain_name][split_rate][slot_name]:
                        self.all_context_dict[domain_name][split_rate][slot_name][context_str].add(slot_value_str)
                    else:
                        self.all_context_dict[domain_name][split_rate][slot_name][context_str] = set()
                        self.all_context_dict[domain_name][split_rate][slot_name][context_str].add(slot_value_str)

                    # ========== update all_temp_dict =========
                    if user_temp not in self.all_temp_dict[domain_name]:
                        self.all_temp_dict[domain_name][user_temp] = {}
                    if slot_name in self.all_temp_dict[domain_name][user_temp]:
                        self.all_temp_dict[domain_name][user_temp][slot_name].add(slot_value_str)
                    else:
                        self.all_temp_dict[domain_name][user_temp][slot_name] = set()
                        self.all_temp_dict[domain_name][user_temp][slot_name].add(slot_value_str)

                    # ========== update train_full_dict = =========
                    if slot_name not in self.train_full_dict[domain_name][split_rate]:
                        self.train_full_dict[domain_name][split_rate][slot_name] = set()
                        self.train_full_dict[domain_name][split_rate][slot_name].add(slot_value_str)
                    else:
                        self.train_full_dict[domain_name][split_rate][slot_name].add(slot_value_str)

                    # if slot_name in self.train_full_dict[domain_name]:
                    #     self.train_full_dict[domain_name][slot_name].add(slot_value_str)
                    # else:
                    #     self.train_full_dict[domain_name][slot_name] = set()
                    #     self.train_full_dict[domain_name][slot_name].add(slot_value_str)

                    # ========= update all_full_dict ==========
                    if slot_name in self.all_full_dict[domain_name]:
                        self.all_full_dict[domain_name][slot_name].add(slot_value_str)
                    else:
                        self.all_full_dict[domain_name][slot_name] = set()
                        self.all_full_dict[domain_name][slot_name].add(slot_value_str)

                elif data_label in ['dev', 'test']:  # record slot name-value case in dev and test set
                    # ========= update all_full_dict ==========
                    if slot_name in self.all_full_dict[domain_name]:
                        self.all_full_dict[domain_name][slot_name].add(slot_value_str)
                    else:
                        self.all_full_dict[domain_name][slot_name] = set()
                        self.all_full_dict[domain_name][slot_name].add(slot_value_str)
                else:
                    print('Error: wrong data label', data_label)
                    raise RuntimeError

    def unpack_and_cook_raw_data(self, raw_data, domain_name):
        # empty the all_data_item pool for current data
        for dm in self.all_domain:
            self.all_data_item[dm] = []
        for dialogue in raw_data:
            word_label_pair_lst = dialogue.split('\n')
            # Store info for a dialogue
            all_user_word_lst = []
            all_slot_label_lst = []
            all_intent_lst = []
            # Store info for one sentence of a dialogue
            user_word_lst = []
            slot_label_lst = []
            intent_lst = []

            for pair in word_label_pair_lst:
                word, label = pair.split()
                word = word.lower()
                if word in ['intent1', 'intent2', 'intent3', ]:
                    if label != 'O':
                        intent_lst.append(label)
                elif word in ['intent4', 'intent5', 'intent6']:  # deal with special case
                    all_intent_lst[-1].append(label)
                else:
                    user_word_lst.append(word)
                    slot_label_lst.append(label)
                # check sentence end
                if word in ['intent3', ]:
                    all_user_word_lst.append(user_word_lst)
                    all_slot_label_lst.append(slot_label_lst)
                    all_intent_lst.append(intent_lst)

                    user_word_lst = []
                    slot_label_lst = []
                    intent_lst = []

            # ======= collecting remained sentences and adapt to ATIS data =======
            all_user_word_lst.append(user_word_lst)
            all_slot_label_lst.append(slot_label_lst)
            all_intent_lst.append(intent_lst)

            for user_word_lst, slot_label_lst, intent_lst in zip(all_user_word_lst, all_slot_label_lst, all_intent_lst):
                data_item = {
                    'user_say': ' '.join(user_word_lst),
                    'user_word_lst': user_word_lst,
                    'user_temp': '',
                    'user_temp_word_lst': [],
                    'label_lst': slot_label_lst,  # A list of name-word, '' represent for empty element
                    'intent_lst': intent_lst,  # A list of name-word, '' represent for empty element
                    'slot_name_lst': [],  # A list of name-word, '' represent for empty element
                    'slot_value_lst': [],  # A list of value-words' list, [''] represent for empty element
                    'context_lst': []  # A list of context-words; list, [''] represent for empty element
                }
                data_item = self.entity_replace(data_item)
                self.all_data_item[domain_name].append(data_item)

    def clustering(self, target_file, split_rate_lst, cluster_mode=None):
        cluster_mode = cluster_mode if cluster_mode else self.cluster_mode
        # processing data
        domain_name = target_file.split('_')[0] + '_' + self.special_mark  # eg: weather_labeled
        data_label = target_file.split('_')[1]  # eg: dev, train, test
        raw_data = self.load_data(self.input_dir + target_file)
        # print('debug!!!', len(self.all_data_item[domain_name]), target_file)
        self.unpack_and_cook_raw_data(raw_data, domain_name)
        print('debug!!!', len(self.all_data_item[domain_name]), target_file)
        # ======= split the data to smaller parts ========
        data_item_set_lst = []  # store different size of data_item set
        if split_rate_lst and 'train' in data_label:
            for split_rate in split_rate_lst:
                if SENT_COUNT_SPLIT:
                    end_index = split_rate
                else:
                    end_index = int(len(self.all_data_item[domain_name]) * split_rate)
                # print('SR and Sentence Count:', split_rate, end_index, domain_name)
                data_item_set_lst.append(self.all_data_item[domain_name][:end_index])
        else:
            data_item_set_lst = [self.all_data_item[domain_name]]
            split_rate_lst = [1]

        # ======= fill the dict for temp refilling =========
        for ind, data_item_set in enumerate(data_item_set_lst):
            for data_item in data_item_set:
                self.update_dict(domain_name, data_item, data_label, split_rate_lst[ind])

        # ======= start clustering with different algorithm =========
        for ind, data_item_set in enumerate(data_item_set_lst):
            print('Start %s clustering by %s on split rate of %f' % (target_file, cluster_mode, split_rate_lst[ind]))
            if cluster_mode == 'all' or cluster_mode == 'slot':
                self.cluster_by_slot(domain_name, data_item_set, data_label, split_rate_lst[ind])
            if cluster_mode == 'all' or cluster_mode == 'intent':
                self.cluster_by_intent(domain_name, data_item_set, data_label, split_rate_lst[ind])
            if cluster_mode == 'all' or cluster_mode == 'slot-intent':
                self.cluster_by_intent_and_slot(domain_name, data_item_set, data_label, split_rate_lst[ind])
            if cluster_mode == 'no_clustering':
                self.no_clustering(domain_name, data_item_set, data_label, split_rate_lst[ind])

    def cluster_by_intent(self, domain_name, data_item_set, data_label, split_rate):
        # cluster and output results
        clustered_data = {}  # clustering data here
        for data_item in data_item_set:
            common_intent = '-'.join(sorted(set(data_item['intent_lst'])))
            if common_intent in clustered_data:
                clustered_data[common_intent].append(data_item)
            else:
                clustered_data[common_intent] = [data_item]

        with open(self.result_dir + data_label + '_' + domain_name + '_' + 'intent' + str(split_rate) + '.json', 'w') as writer:
            json.dump(clustered_data, writer, indent=2)

    def cluster_by_slot(self, domain_name, data_item_set, data_label, split_rate):
        # cluster and output results
        clustered_data = {}  # clustering data here
        for data_item in data_item_set:
            # print('=====================', data_item_set)
            common_slot = '-'.join(sorted(set(data_item['slot_name_lst'])))
            if common_slot in clustered_data:
                clustered_data[common_slot].append(data_item)
            else:
                clustered_data[common_slot] = [data_item]
        with open(self.result_dir + data_label + '_' + domain_name + '_' + 'slot' + str(split_rate) + '.json', 'w') as writer:
            json.dump(clustered_data, writer, indent=2)

    def cluster_by_intent_and_slot(self, domain_name, data_item_set, data_label, split_rate):
        # cluster and output results
        clustered_data = {}  # clustering data here
        for data_item in data_item_set:
            common_intent_and_slot = '-'.join(sorted(set(data_item['slot_name_lst'] + data_item['intent_lst'])))
            if common_intent_and_slot in clustered_data:
                clustered_data[common_intent_and_slot].append(data_item)
            else:
                clustered_data[common_intent_and_slot] = [data_item]
        with open(self.result_dir + data_label + '_' + domain_name + '_' + 'intent-slot' + str(split_rate) + '.json', 'w') as writer:
            json.dump(clustered_data, writer, indent=2)

    def no_clustering(self, domain_name, data_item_set, data_label, split_rate):
        # don't cluster and output results
        clustered_data = {}  # clustering data here
        for data_item in data_item_set:
            common_value = ''
            if common_value in clustered_data:
                clustered_data[common_value].append(data_item)
            else:
                clustered_data[common_value] = [data_item]
        with open(self.result_dir + data_label + '_' + domain_name + '_' + 'nc' + str(split_rate) + '.json',
                  'w') as writer:
            json.dump(clustered_data, writer, indent=2)

    def dump_dict(self, split_rate_lst):
        self.deep_change_set_to_list_dict(self.all_context_dict)
        self.deep_change_set_to_list_dict(self.all_temp_dict)
        self.deep_change_set_to_list_dict(self.all_full_dict)
        self.deep_change_set_to_list_dict(self.train_full_dict)

        for domain_name in self.all_domain:
            for split_rate in split_rate_lst:
                with open(self.result_dir + domain_name + str(split_rate) + '_context-query.dict', 'w') as writer:
                    json.dump(self.all_context_dict[domain_name][split_rate], writer, indent=2)
                with open(self.result_dir + domain_name + str(split_rate) + '_train_full-query.dict', 'w') as writer:
                    json.dump(self.train_full_dict[domain_name][split_rate], writer, indent=2)
            with open(self.result_dir + domain_name + '_temp-query.dict', 'w') as writer:
                json.dump(self.all_temp_dict[domain_name], writer, indent=2)
            with open(self.result_dir + domain_name + '_full-query.dict', 'w') as writer:
                json.dump(self.all_full_dict[domain_name], writer, indent=2)
            # with open(self.result_dir + domain_name + '_train_full-query.dict', 'w') as writer:
            #     json.dump(self.train_full_dict[domain_name], writer, indent=2)

    def deep_change_set_to_list_dict(self, d):
        for key in d:
            if type(d[key]) == dict:
                self.deep_change_set_to_list_dict(d[key])
            elif type(d[key]) == set:
                d[key] = list(d[key])

    @staticmethod
    def load_data(target_path):
        with open(target_path, 'r') as reader:
            return reader.read().strip().split('\n\n')

    @staticmethod
    def entity_replace(data_item):
        """
        Replace slot value with slot name in the template.
        Notice:
            Element lists, including slot_name, slot_value and context_lst, are aligned with template word list.
            And '' is used as padding element.
        :param data_item: Dict type, saving different property
        :return:
        """
        word_lst = data_item['user_word_lst']
        label_lst = data_item['label_lst']
        temp_word_lst = []
        slot_name_lst = []
        slot_value_lst = []
        context_lst = []
        for w, l in zip(word_lst, label_lst):
            if l == 'O':
                temp_word_lst.append(w)
                slot_name_lst.append('')
                slot_value_lst.append([''])
            elif 'B-' in l:
                slot_name = l.replace('B-', '')
                temp_word_lst.append('<%s>' % slot_name)
                slot_name_lst.append(slot_name)
                slot_value_lst.append([w])
            elif 'I-' in l:
                slot_value_lst[-1].append(w)

        for ind, pair in enumerate(zip(temp_word_lst, slot_name_lst, slot_value_lst)):
            temp_w, slot_name, slot_value = pair
            if slot_name:
                context_text = temp_word_lst[max(ind - CONTEXT_WINDOW_SIZE, 0): ind + CONTEXT_WINDOW_SIZE + 1]
            else:
                context_text = ['']
            context_lst.append(context_text)

        data_item['user_temp_word_lst'] = temp_word_lst
        data_item['user_temp'] = ' '.join(temp_word_lst)
        data_item['slot_name_lst'] = slot_name_lst
        data_item['slot_value_lst'] = slot_value_lst
        data_item['context_lst'] = context_lst
        return data_item


def clustering_and_dump_dict(data_dir, config=None, cluster_mode='all', train_set_split_rate_lst=None, special_mark='labeled'):
    print('user utterance clustering')
    if not config:
        with open('../../config.json', 'r') as con_f:
            config = json.load(con_f)
    all_file = os.listdir(data_dir)

    # =========== collect domain name ==========
    all_domain = set()
    print(all_file)
    for file_name in all_file:
        all_domain.add(str(file_name.split('_')[0]) + '_' + special_mark)

    tmp_cluster = Cluster(
        input_dir=data_dir,
        result_dir=config['path']['ClusteringResult'],
        all_domain=all_domain,
        cluster_mode=cluster_mode,
        special_mark=special_mark,
    )
    for f in all_file:
        tmp_cluster.clustering(f, train_set_split_rate_lst)

    tmp_cluster.dump_dict(train_set_split_rate_lst)
