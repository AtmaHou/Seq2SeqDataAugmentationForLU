# coding: utf-8
from __future__ import unicode_literals, print_function, division
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
MAX_LENGTH = 10


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def indexes_from_sentence(word_table, sentence):
    return [word_table.word2index[word] for word in sentence]


def variable_from_sentence(word_table, sentence):
    indexes = indexes_from_sentence(word_table, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result


def variables_from_pair(pair, input_word_table, output_word_table):
    input_variable = variable_from_sentence(input_word_table, pair[0])
    target_variable = variable_from_sentence(output_word_table, pair[1])
    return input_variable, target_variable
