import argparse
import codecs
import os
import re
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('gold_file', type=str, metavar='', help='')
parser.add_argument('system_file', type=str, metavar='', help='')

A = parser.parse_args()

def read_425():
    f = open('../../processed/425_tags')
    tag_list = []
    for line in f:
        args = line.strip().split('\t')
        tag_list.append(args[0])
    f.close()
    return tag_list

if __name__ == '__main__':

    predefined_tags = read_425()
    gold_tags = []
    f = open(A.gold_file, "r")
    for line in f:
        line = line.strip()
        # tags = [x.split("|")[2] if x.split("|")[2] in predefined_tags else "<DUMMY_SUPERTAG>" for x in line.split(" ") ]
        tags = [x.split("|")[2] for x in line.split(" ") ]
        gold_tags.append(tags)
    f.close()

    system_tags = []
    f = open(A.system_file, "r")
    for line in f:
        line = line.strip()
        tags = line.split(" ")
        system_tags.append(tags)
    f.close()
    assert(len(system_tags) == len(gold_tags))
    cor = 0
    tot = 0
    for i in xrange(0, len(system_tags)):
        assert(len(gold_tags[i]) == len(system_tags[i]))
        for j in xrange(0, len(gold_tags[i])):
            if gold_tags[i][j] == system_tags[i][j]:
                cor += 1
            tot += 1
    print (float)(cor) / (float)(tot)
