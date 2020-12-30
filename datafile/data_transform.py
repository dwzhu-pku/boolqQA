import json
from nltk.tokenize import wordpunct_tokenize

''' train '''
# f = open("/Users/gongbeida/Documents/GitHub/boolqQA/datafile/train.jsonl", 'r')
# path1 = './train_sen.txt'
# path2 = './train_label.txt'
''' val '''
f = open("/Users/gongbeida/Documents/GitHub/boolqQA/datafile/dev.jsonl", 'r')
path1 = './val_sen.txt'
path2 = './val_label.txt'

fw1 = open(path1, 'w')
fw2 = open(path2, 'w')

maxL1 = 0
maxL2 = 0

for lines in f.readlines():
    lines = json.loads(lines)
    sen1 = lines['question']
    sen2 = lines['passage']
    ans = lines['answer']
    # cur_len1 = len(wordpunct_tokenize(sen1))
    # maxL1 = max(maxL1, cur_len1)
    # cur_len2 = len(wordpunct_tokenize(sen2))
    # maxL2 = max(maxL2, cur_len2)
    fw1.write(sen1 + '\n')
    fw1.write(sen2 + '\n')
    if ans:
        fw2.write('1\n')
    else: 
        fw2.write('0\n')
print(maxL1, maxL2)
