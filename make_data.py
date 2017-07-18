#coding: utf-8
import random
f = open("relation_1")
f_a = open("data", "w")
line = f.readline()
dic = {}
while line:
	words = (line.strip("\n")).split(" ")
	if words[0] in dic.keys():
		dic[words[0]].append(words[1])
	else:
		dic[words[0]] = [words[1]]
	line = f.readline()
keys = dic.keys()
random.shuffle(keys)
i = 0
for key in keys:
	i += 1
	random.shuffle(dic[key])
	print key + " " + dic[key][0]
	f_a.write(key + " " + dic[key][0] + "\n")
	#print i
	if i == 500:
		break
