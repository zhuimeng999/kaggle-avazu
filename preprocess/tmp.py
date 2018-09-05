import sys
print(sys.argv)
with open(sys.argv[1], 'r') as fin, open(sys.argv[2], 'w') as fout:
    for line in fin:
        fout.write(' \t'.join(line.split(' ')))