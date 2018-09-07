import os

'''
-s <type> : Type of machine learning model (default 0)
   for classification task:
       0 -- linear model (GLM)
       1 -- factorization machines (FM)
       2 -- field-aware factorization machines (FFM)
   for regression task:
       3 -- linear model (GLM)
       4 -- factorization machines (FM)
       5 -- field-aware factorization machines (FFM)
'''

if __name__ == '__main__':
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(curr_dir)
    data_train = os.path.join(curr_dir, '../../preprocess/output/trainsvm_raw_0.libsvm')
    data_valid = os.path.join(curr_dir, '../../preprocess/output/validsvm_raw_0.libsvm')
    model_path = os.path.join(curr_dir, 'model.model')
    cmd = os.path.join(curr_dir, '../../bin/xlearn_train ')
    cmd = cmd + data_train + ' '
    cmd = cmd + '-v ' + data_valid + ' '
    cmd = cmd + '-m ' + model_path + ' '
    cmd = cmd + '-s 0 '
    cmd = cmd + '-p ftrl -alpha 0.002 -beta 0.8 -lambda_1 0.001 -lambda_2 1.0' + ' '# ftrl sgd adagrad
    cmd = cmd + '-r 0.002' + ' '
    cmd = cmd + '-e 100' + ' '
    print(cmd)
    os.system(cmd)
