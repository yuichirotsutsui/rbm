#coding:utf-8

import GBRBM
import numpy
import os
import sys
import pickle
import random
import copy
import cupy
import chainer.cuda
import chainer
from chainer import computational_graph
from chainer import cuda, Variable

def load_vec_data(file_name):
    if os.path.exists(file_name):
        f = open(file_name)
    else:
        print "ファイルがありません"
        exit()
    line = f.readline()
    array1 = []
    array2 = []
    while line:
        line1 = line.strip(" ").split(" ")
        line = f.readline()
        line2 = line.strip("\n").split(" ")
        line = f.readline()
        array1.append(map(float, line1))
        array2.append(map(float, line2))
    data1 = numpy.array(array1)
    data2 = numpy.array(array2)
    return data1,len(array1[0]), data2, len(array2[0])

def load_dic(dic_path):
    dic = {}
    if os.path.exists(dic_path):
         f = open("renso_normalized")
    else:
        print "ファイルがありません"
        exit()
    line = f.readline()
    while line:
        word = line.strip("\n").split(" ")[0]
        vecs = line.strip("\n").split(" ")[1:]
        dic[word] = vecs
        line = f.readline()
    f.close()
    return dic

def learn_rbm(rbm, learning_rate = 0.0001, k = 1, training_epochs = 1000, batch_size = 10):
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr = learning_rate, k = k, batch_size = batch_size)
        cost = rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost

def get_ans_gbrbm(data, gbrbm, reconstruct_num = 100):
    zero_data = numpy.array([[0 for i in range(200)] for j in range(len(data))])
    data_left = numpy.hsplit(data, [200])[0]
    data2 = gbrbm.make_memory(numpy.c_[data_left, zero_data], num = reconstruct_num, data = data_left, var = 0.00000000001, simple_sample = True)
    return numpy.hsplit(data2, [200])[1]

def load_test_vec(file_name1, file_name2):
    if os.path.exists(file_name1) and os.path.exists(file_name2):
        f1 = open(file_name1)
        f2 = open(file_name2)
    else:
        return 0
    dic = {}
    line1 = f1.readline()
    line2 = f2.readline()
    while line1:
        tmp = (line1.strip("\n")).split(" ")
        for j in range(2):
            if tmp[j * 2] in dic:
                pass
            else:
                array = []
                tmp2 = (line2.strip("\n")).split(" ")
                for i in range(len(tmp2)):
                    array.append(tmp2[i])
                dic[tmp[j * 2]] = numpy.array(array)
            line2 = f2.readline()
            line2 = f2.readline()
        line1 = f1.readline()
    return dic

def calc_ans(file_name, data, dic, top = 1, print_result = False, file_num = 0):
    count = 0.0
    ans_num = 0.0
    if os.path.exists(file_name):
        f = open(file_name)
    word = dic.keys()
    if print_result == True:
        ff = open("./ans/ans_" + str(file_num), "w")
    for i in range(len(data)):
        line = f.readline()
        num_list = []
        word_list = []
        for j in range(len(word)):
            tmp = []
            for k in range(len(dic[word[j]])):
                tmp.append(float(dic[word[j]][k]))
            num = numpy.dot(data[i], numpy.array(tmp)) / (numpy.linalg.norm(data[i]) * numpy.linalg.norm(numpy.array(tmp)))
            check = 0
            for k in range(len(num_list)):
                if num_list[k] < num:
                    num_list.insert(k, num)
                    word_list.insert(k, word[j])
                    check = 1
                    break
            if check == 0:
                num_list.append(num)
                word_list.append(word[j])
        for j in range(top):
            count += 1.0
            if (line.strip("\n")).split(" ")[1] == word_list[j]:
                ans_num += 1.0
            print line.strip("\n")
            print str(j) + ":" + word_list[j] + ":" + str(num_list[j])
            print str(100 * ans_num / count) + "\n"
            if print_result == True:
                ff.write(line.strip("\n") + "\n")
                ff.write(str(j) + ":" + word_list[j] + ":" + str(num_list[j]) + "\n")
                ff.write(str(100 * ans_num / count) + "\n")
    if print_result == True:
        ff.close()

load_gbrbm = False
pp = 0.0001
hidden_layer = 100
epoch = 1
learning_file = "relation_3"
dic_path = "./renso_normalized"
reconstruct_num = 100
batch_size = 5

rng = cupy.random.RandomState(123)
data1,data1_len, data2, data2_len = load_vec_data("./" + learning_file + "_dir/" + learning_file + "_vec")
dic = load_dic(dic_path)
learning_data = numpy.c_[data1, data2]
learning_data_len = data1_len + data2_len

gbrbm = None
if load_gbrbm:
    f_gbrbm = open("gbrbm_dump")
    gbrbm = pickle.load(f_gbrbm)
    f_gbrbm.close()
else:
    gbrbm = GBRBM.GBRBM(input = copy.copy(learning_data), n_visible = learning_data_len, n_hidden = hidden_layer, cupy_rng = rng)

learn_rbm(gbrbm, training_epochs = epoch, learning_rate = pp, batch_size = batch_size)
f_gbrbm = open("gbrbm_dump", "w")
pickle.dump(gbrbm, f_gbrbm)
f_gbrbm.close()
ans = get_ans_gbrbm(learning_data, gbrbm, reconstruct_num = reconstruct_num)
calc_ans(learning_file, chainer.cuda.to_cpu(ans), dic, top = 1, print_result = False, file_num = 0)
