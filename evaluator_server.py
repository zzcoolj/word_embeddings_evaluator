import evaluator
import pandas as pd
import os

import sys
sys.path.insert(0, '../common/')
import multi_processing

'''
Evaluate results in matrix2vec/output/vectors/ppmi_svd
Evaluate results in matrix2vec/output/vectors/firstOrder_svd
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


# evaluate_folder(folder_path='../matrix2vec/output/vectors/ppmi_svd/', file_prefix='ppmi_svd_')
evaluate_folder(folder_path='../matrix2vec/output/vectors/firstOrder_svd/', file_prefix='firstOrder_svd_')