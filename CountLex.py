import argparse
import codecs
import os
import re
import sys

parser = argparse.ArgumentParser(description='')

A = parser.parse_args()

def build_taglist(f):
    tag_dict = dict([])
    for line in f:
        line = line.strip()
        args = re.split(' +', line)
        # print args[1], int(args[2])
        if args[1] in tag_dict:
            tag_dict[args[1]] += int(args[2])
        else:
            tag_dict[args[1]] = int(args[2])
    tag_list = tag_dict.items()
    tag_list = sorted(tag_list, key = lambda i : i[1], reverse = True)
    print "sum of token: " + str(sum([i[1] for i in tag_list]))
    return tag_list

def extract_425():
    f = open('../ccgbank_1_1/data/LEX/CCGbank.02-21.lexicon')
    tag_list = build_taglist(f)
    for i in tag_list:
        if i[1] < 10:
            continue
        print i[0] + "\t" + str(i[1])
    f.close()

def read_425():
    f = open('../processed/425_tags')
    tag_list = []
    for line in f:
        args = line.strip().split('\t')
        tag_list.append(args[0])
    f.close()
    return tag_list

if __name__ == '__main__':
    f = open('../ccgbank_1_1/data/LEX/CCGbank.02-21.lexicon')
    tag_list_dev = build_taglist(f)
    f.close()

    tag_425 = read_425()

    count_in = 0
    count_out = 0

    dev_tags = [i[0] for i in tag_list_dev]

    print "tags number in dev: " + str(len(dev_tags))

    seen = 0
    for tag in dev_tags:
        if tag in tag_425:
            seen += 1
        else:
            print tag
    print "tags never seen in training: " + str(len(dev_tags) - seen)


    for i in tag_list_dev:
        tag = i[0]
        count = i[1]
        if tag in tag_425:
            count_in += count
        else:
            count_out += count
    print count_in, count_out, (float(count_out)/(count_in + count_out))
