import numpy as np
import pandas as pd

from modules import fasta_reader, l_gram_freq, metrics, pca

window = 20  # размер окна
step = 5  # шаг окна
L = 1  # размер L-граммы

# используемые метрики между траекториями (из модуля metrics)
metrics_list = [
    # 'mean',
    'hausdorff',
    # 'frechet',
    # 'jaccard_tr'
]


def prepare_data():
    data = pd.read_csv('data/tmppred.csv')

    mesophilic = data[data['Type'] == 'mesophilic protein']
    thermophilic = data[data['Type'] == 'thermophilic protein']
    print('mesophilic count:', mesophilic.shape[0])
    print('thermophilic count:', thermophilic.shape[0])

    all_seqs = []
    seq_lens = []

    for _, row in mesophilic.iterrows():
        seqs = fasta_reader.read_seqs_from_str(row['Sequence'], window, step)
        all_seqs.extend(seqs)
        seq_lens.append(len(seqs))

    for _, row in thermophilic.iterrows():
        seqs = fasta_reader.read_seqs_from_str(row['Sequence'], window, step)
        all_seqs.extend(seqs)
        seq_lens.append(len(seqs))

    # считаем частоты L-грамм
    freq_list = l_gram_freq.calculate_frequencies(all_seqs, L)
    # считаем расстояния между фрагментами, используем метрику Пифагора
    dists = metrics.pythagoras(np.array(freq_list))
    # считаем главные компоненты фрагментов
    dispersions, pr_comps = pca.pca(dists)

    # строим фазовые траектории
    trajectories = []
    i = 0
    for seq_len in seq_lens:
        trajectories.append(pr_comps[i:i + seq_len])
        i += seq_len
    trajectories = np.array(trajectories, dtype=object)

    # считаем главные компоненты последовательностей на основе расстояний между траекториями,
    # посчитанных различными метриками, сохраняем их в папку 'data/precalculated'
    for metric in metrics_list:
        metrics_function = getattr(metrics, metric)
        dists = metrics_function(trajectories)
        dispersions, pr_comps = pca.pca(dists)
        np.save('data/precalculated/' + metric, pr_comps)
