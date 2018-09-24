import evaluator
import pandas as pd
import os

import sys
sys.path.insert(0, '../common/')
import multi_processing

'''
Evaluate results in matrix2vec/output/vectors/ppmi_svd
Evaluate results in matrix2vec/output/vectors/firstOrder_svd
Evaluate results in matrix2vec/output/vectors/rw0_svd
'''


def evaluate_folder(folder_path, file_prefix):
    df = pd.DataFrame(columns=[
                # word embeddings file name
                'file name',
                # wordsim353
                'wordsim353_Pearson correlation', 'Pearson pvalue',
                'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
                # simlex999
                'simlex999_Pearson correlation', 'Pearson pvalue',
                'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
                # questions-words
                'sem_acc', '#sem', 'syn_acc', '#syn', 'total_acc', '#total'
            ])
    i = 0

    for w in range(2, 11):
        e = evaluator.Evaluator.from_storage(
            tokens_path='../matrix2vec/input/encoded_edges_count_window_size_' + str(w) + '_undirected_tokens.pickle')
        for d in [200, 500, 800, 1000]:
            file_name = file_prefix + 'w' + str(w) + '_d' + str(d)
            result = e.evaluate(folder_path + file_name + '.npy', matrix_type='npy')
            df.loc[i] = [file_name] + result
            print([file_name] + result)
            i += 1

    writer = pd.ExcelWriter(folder_path + 'result.xlsx')
    df.to_excel(writer, 'Sheet1')
    writer.save()

    # Delete all _embeddings.csv files
    files_to_delete = multi_processing.get_files_endswith(data_folder=folder_path,
                                                          file_extension='_embeddings.csv')
    for file_path in files_to_delete:
        print('Remove file %s' % file_path)
        os.remove(file_path)


# evaluate_folder(folder_path='../matrix2vec/output/vectors/cooc_normalized_svd/', file_prefix='cooc_normalized_svd_')
evaluate_folder(folder_path='../matrix2vec/output/vectors/cooc_normalized_smoothed_svd/', file_prefix='cooc_normalized_smoothed_svd_')
# evaluate_folder(folder_path='../matrix2vec/output/vectors/firstOrder_normalized_svd/', file_prefix='firstOrder_normalized_svd_')
# evaluate_folder(folder_path='../matrix2vec/output/vectors/firstOrder_normalized_smoothed_svd/', file_prefix='firstOrder_normalized_smoothed_svd_')
# evaluate_folder(folder_path='../matrix2vec/output/vectors/rw0_normalized_svd/', file_prefix='rw0_normalized_svd_')
# evaluate_folder(folder_path='../matrix2vec/output/vectors/rw1_normalized_svd/', file_prefix='rw1_normalized_svd_')
# evaluate_folder(folder_path='../matrix2vec/output/vectors/rw2_normalized_svd/', file_prefix='rw2_normalized_svd_')


"""
Evaluate results in ppmi+firstOrder_svd
Evaluate results in ppmi+rw1_svd
"""


def evaluate_folder_bis(folder_path, window_size, file_name_prefix):
    df = pd.DataFrame(columns=[
                # word embeddings file name
                'file name',
                # wordsim353
                'wordsim353_Pearson correlation', 'Pearson pvalue',
                'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
                # simlex999
                'simlex999_Pearson correlation', 'Pearson pvalue',
                'Spearman correlation', 'Spearman pvalue', 'Ration of pairs with OOV',
                # questions-words
                'sem_acc', '#sem', 'syn_acc', '#syn', 'total_acc', '#total'
            ])
    i = 0

    e = evaluator.Evaluator.from_storage(
        tokens_path='../matrix2vec/input/encoded_edges_count_window_size_' + str(window_size) + '_undirected_tokens.pickle')
    for k in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, -0.1, -0.2, -0.5, -1, -2, -5, -10, -20, -50, -100]:
        for dimension in [500, 800, 1000]:
            file_name = file_name_prefix+str(k)+'_svd_d' + str(dimension)
            result = e.evaluate(folder_path + file_name + '.npy', matrix_type='npy')
            df.loc[i] = [file_name] + result
            print([file_name] + result)
            i += 1

    writer = pd.ExcelWriter(folder_path + 'result.xlsx')
    df.to_excel(writer, 'Sheet1')
    writer.save()

    # Delete all _embeddings.csv files
    files_to_delete = multi_processing.get_files_endswith(data_folder=folder_path,
                                                          file_extension='_embeddings.csv')
    for file_path in files_to_delete:
        print('Remove file %s' % file_path)
        os.remove(file_path)


# evaluate_folder_bis(folder_path='../matrix2vec/output/vectors/ppmi+firstOrder_svd/', window_size=5, file_name_prefix='ppmi_w5_+firstOrder_w5_k')
# evaluate_folder_bis(folder_path='../matrix2vec/output/vectors/ppmi+rw1_svd/', window_size=5, file_name_prefix='ppmi_w5_+rw1_w7_k')
# evaluate_folder_bis(folder_path='../matrix2vec/output/vectors/ppmi+rw2_svd/', window_size=5, file_name_prefix='ppmi_w5_+rw2_w3_k')
# evaluate_folder_bis(folder_path='../matrix2vec/output/vectors/cooc_firstOrder_normalized_svd/', window_size=5, file_name_prefix='cooc_w5_firstOrder_w5_normalized_k')

'''
specific
'''

e = evaluator.Evaluator.from_storage(tokens_path='../matrix2vec/input/encoded_edges_count_window_size_5_undirected_tokens.pickle')
# print(e.evaluate(matrix_path='../matrix2vec/output/vectors/specific/specific_k-1_svd_d500.npy', matrix_type='npy'))
# print(e.evaluate(matrix_path='../matrix2vec/output/vectors/specific/specific_k-1_svd_d800.npy', matrix_type='npy'))
# print(e.evaluate(matrix_path='../matrix2vec/output/vectors/specific/specific_k-1_svd_d1000.npy', matrix_type='npy'))
print(e.evaluate(matrix_path='../matrix2vec/output/vectors/specific/test.npy', matrix_type='npy'))