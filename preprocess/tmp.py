import sys, os
dirname = 'D:\\lucius_files\\learn\\usif\\kaggle-avazu\\preprocess\\output'
base_path = os.path.join(dirname, 'train_format.csv')
train_path = os.path.join(dirname, 'train_raw_0.csv')
valid_path = os.path.join(dirname, 'valid_raw_0.csv')
with open(base_path, 'r') as fin:
    with open(train_path, 'w') as ft, open(valid_path, 'w') as fv:
        header = fin.readline()
        ft.write(header)
        fv.write(header)
        for line_no, line in enumerate(fin):
            if line_no < 5000:
                ft.write(line)
            elif line_no < 10000:
                fv.write(line)
            else:
                break