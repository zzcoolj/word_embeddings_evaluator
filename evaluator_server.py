import evaluator

'''
Evaluate results in matrix2vec/output/vectors/ppmi_svd
'''

for w in range(2, 11):
    e = evaluator.Evaluator.from_storage(
        tokens_path='../matrix2vec/input/encoded_edges_count_window_size_' + str(w) + '_undirected_tokens.pickle')
    for d in [200, 500, 800, 1000]:
        result = e.evaluate('ppmi_svd_w' + str(w) + '_d' + str(d) + '.npy', matrix_type='npy')
        print(result)
        exit()
