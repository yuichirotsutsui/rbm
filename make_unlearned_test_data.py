#coding: utf-8
import sys
import os
import shutil

argvs = sys.argv
argc = len(argvs)

if argc != 2:
  print "変換したいファイルを指定してください"
  exit()
  

dic = {}
f = open("renso_normalized")
line = f.readline()
while line:
  word = line.strip("\n").split(" ")[0]
  vecs = line.strip("\n").split(" ")[1:]
  dic[word] = vecs
  line = f.readline()
f.close()

f = open(argvs[1])
line = f.readline()
if os.path.exists(argvs[1] + "_dir"):
  shutil.rmtree(argvs[1] + "_dir")
os.mkdir(argvs[1] + "_dir")
f_w = open("./" + argvs[1] + "_dir/" + argvs[1] + "_vec", "w")
f_w_n = open("./" + argvs[1] + "_dir/" + argvs[1] + "_not", "w")
f_w_u = open("./" + argvs[1] + "_dir/" + argvs[1] + "_word", "w")

while line:
  word = line.strip("\n")
  if word in dic:
    f_w.write(" ".join(dic[word]) + "\n")
    f_w_u.write(line)
  else:
    f_w_n.write(word + "\n")
  line = f.readline()

f.close()
f_w.close()
f_w_u.close()
f_w_n.close()
