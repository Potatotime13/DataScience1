import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    results = load_results()
    width = 0.20
    x = np.arange(4)
    plt.bar(x-width, results[0][0].values[:, 1], width)
    plt.bar(x, results[1][0].values[:, 1], width)
    plt.xticks(x, results[0][0].values[:, 0])
    plt.legend(["user/user", "ncf"])
    plt.title('MovieLens')
    plt.show()

    print()


def load_results():
    result_user = []
    result_user_book = []
    result_ncf = []
    for i in range(4):
        result_user.append(pd.read_csv('perf/person_user_' + str(i) + '.csv'))
        result_user_book.append(pd.read_csv('perf/person_user_book_' + str(i) + '.csv'))
        result_ncf.append(pd.read_csv('perf/ncf_mov_' + str(i) + '.csv'))
    return result_user, result_ncf, result_user_book


if __name__ == "__main__":
    main()