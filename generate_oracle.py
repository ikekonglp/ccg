import argparse
import codecs
import os
import re
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('gold_file', type=str, metavar='', help='')
A = parser.parse_args()

# ["S[dcl]", None, None]
def is_leaf(tag):
    assert(len(tag) == 3)
    if tag[1] is None and tag[2] is None:
        return True
    return False

def split_tag(current_tag):
    if current_tag[0] == "(":
        stack = []
        stack.append("(")
        split_point = -1
        for i in xrange(1, len(current_tag)):
            if current_tag[i] == ")":
                stack.pop()
            elif current_tag[i] == "(":
                stack.append("(")
            if len(stack) == 0:
                # we are one position before the split point
                split_point = (i+1)
                break
        if split_point == len(current_tag):
            # a leaf node already
            return split_tag(current_tag[1:(len(current_tag)-1)])


        # return ["/", "(S[dcl]\NP)", "(S[ng]\NP)"]
        return [current_tag[split_point], split_tag(current_tag[:split_point]), split_tag(current_tag[(split_point+1):])]

    else:
        # start with base category char, so just find the next / or \
        split_point = -1
        for i in xrange(0, len(current_tag)):
            if current_tag[i] == "/" or current_tag[i] == "\\":
                split_point = i
                break
        if split_point == -1:
            # leaf tag
            return current_tag
        else:
            return [current_tag[split_point], split_tag(current_tag[:split_point]), split_tag(current_tag[(split_point+1):])]

def test_parse_tag():
    tags = ["(S[b]\NP)/NP", "(S[dcl]\NP)/(S[ng]\NP)", "N/N"]
    for tag in tags:
        print split_tag(tag)

if __name__ == '__main__':
    f = open(A.gold_file, "r")
    for line in f:
        print line
        line = line.strip()
        # tags = [x.split("|")[2] if x.split("|")[2] in predefined_tags else "<DUMMY_SUPERTAG>" for x in line.split(" ") ]
        tags = [x.split("|")[2] for x in line.split(" ") ]
        for tag in tags:

            print tag + "\t" + str(split_tag(tag))
    f.close()

