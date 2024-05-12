import numpy as np
from sklearn import svm
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

metric = 'hausdorff'
mesophilic_count = 101
thermophilic_count = 105


def classify():
    # загружаем предобученные в data_preprocessor главные компоненты
    pr_comps = np.load('data/precalculated/' + metric + '.npy')
    # используем первые 10 главных компонент для обучения
    X = pr_comps[:, :10]
    y = [0] * mesophilic_count
    y.extend([1] * thermophilic_count)

    # создаем и обучаем бинарный классификатор
    clf = svm.SVC(random_state=0)
    scores = cross_val_score(clf, X, y, cv=RepeatedKFold(random_state=0))
    print('classifier score:', scores.mean())
