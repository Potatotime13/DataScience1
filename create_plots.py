import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_m = load_results_movie()
    results_b = load_results_book()
    width = 0.20

    # movie lens plots
    x = np.arange(len(results_m[0][0].values[:, 0]))
    plt.bar(x-width*1.5, results_m[0][0].values[:, 1], width, color='cyan')
    plt.bar(x-width*0.5, results_m[1][0].values[:, 1], width, color='orange')
    plt.bar(x+width*0.5, results_m[2][0].values[:, 1], width, color='green')
    plt.bar(x+width*1.5, results_m[3][0].values[:, 1], width, color='blue')
    plt.xticks(x, results_m[0][0].values[:, 0])
    plt.legend(["user/user pearson", "user/user euclidean", "ncf",  "item/item pearson"])
    plt.title('MovieLens')
    plt.show()

    x = np.arange(len(results_m[0][1].values[:, 0]))
    plt.bar(x-width*1.5, results_m[0][1].values[:, 1], width, color='cyan')
    plt.bar(x-width*0.5, results_m[1][1].values[:, 1], width, color='orange')
    plt.bar(x+width*0.5, results_m[2][1].values[:, 1], width, color='green')
    plt.bar(x+width*1.5, results_m[3][1].values[:, 1], width, color='blue')
    plt.xticks(x, results_m[0][1].values[:, 0])
    plt.legend(["user/user pearson", "user/user euclidean", "ncf", "item/item pearson"])
    plt.title('MovieLens')
    plt.show()

    x = np.arange(len(results_m[0][2].values[0, 1:]))
    plt.bar(x-width*1.5, results_m[0][2].values[4, 1:], width, color='cyan')
    plt.bar(x-width*0.5, results_m[1][2].values[4, 1:], width, color='orange')
    plt.bar(x+width*0.5, results_m[2][2].values[4, 1:], width, color='green')
    plt.bar(x+width*1.5, results_m[3][2].values[4, 1:], width, color='blue')
    plt.xticks(x, results_m[0][2].values[0, 1:])
    plt.legend(["user/user pearson", "user/user euclidean", "ncf", "item/item pearson"])
    plt.title('MovieLens - MSE distribution')
    plt.show()

    x = np.arange(len(results_m[0][2].values[0, 1:]))
    plt.bar(x-width*1.5, results_m[0][2].values[6, 1:], width, color='cyan')
    plt.bar(x-width*0.5, results_m[1][2].values[6, 1:], width, color='orange')
    plt.bar(x+width*0.5, results_m[2][2].values[6, 1:], width, color='green')
    plt.bar(x+width*1.5, results_m[3][2].values[6, 1:], width, color='blue')
    plt.xticks(x, results_m[0][2].values[0, 1:])
    plt.legend(["user/user pearson", "user/user euclidean", "ncf", "item/item pearson"])
    plt.title('MovieLens - MAE distribution')
    plt.show()

    thr = 2.5
    mask = results_m[0][3].loc[0].values[1:] <= thr
    vals = results_m[0][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4u = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_m[1][3].loc[0].values[1:] <= thr
    vals = results_m[1][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4ue = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_m[2][3].loc[0].values[1:] <= thr
    vals = results_m[2][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4n = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_m[3][3].loc[0].values[1:] <= thr
    vals = results_m[3][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4i = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()

    x = np.arange(1)
    plt.bar(x - width * 1.5, mse_4u, width, color='cyan')
    plt.bar(x - width * 0.5, mse_4ue, width, color='orange')
    plt.bar(x + width * 0.5, mse_4n, width, color='green')
    plt.bar(x + width * 1.5, mse_4i, width, color='blue')
    plt.xticks(x, [''])
    plt.ylabel('average mse')
    plt.legend(["user/user pearson", "user/user euclidean", "ncf", "item/item pearson"])
    plt.title('MovieLens - mse for predictions <'+str(thr))
    plt.show()

    thr = 4.5
    mask = results_m[0][3].loc[0].values[1:] >= thr
    vals = results_m[0][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4u = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_m[1][3].loc[0].values[1:] >= thr
    vals = results_m[1][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4ue = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_m[2][3].loc[0].values[1:] >= thr
    vals = results_m[2][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4n = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_m[3][3].loc[0].values[1:] >= thr
    vals = results_m[3][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4i = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()

    x = np.arange(1)
    plt.bar(x - width * 1.5, mse_4u, width, color='cyan')
    plt.bar(x - width * 0.5, mse_4ue, width, color='orange')
    plt.bar(x + width * 0.5, mse_4n, width, color='green')
    plt.bar(x + width * 1.5, mse_4i, width, color='blue')
    plt.xticks(x, [''])
    plt.ylabel('average mse')
    plt.legend(["user/user pearson", "user/user euclidean", "ncf", "item/item pearson"])
    plt.title('MovieLens - mse for predictions <'+str(thr))
    plt.show()

    # book crossing plots
    x = np.arange(len(results_b[0][0].values[:, 0]))
    plt.bar(x - width, results_b[0][0].values[:, 1], width, color='cyan')
    plt.bar(x, results_b[1][0].values[:, 1], width, color='orange')
    plt.bar(x + width, results_b[2][0].values[:, 1], width, color='blue')
    plt.bar(x + width*2, results_b[3][0].values[:, 1], width, color='red')
    plt.xticks(x, results_b[0][0].values[:, 0])
    plt.legend(["user/user pearson", "user/user euclidean", "item/item pearson", "user / user filtered"])
    plt.title('Book-Crossing')
    plt.show()

    x = np.arange(len(results_b[0][1].values[:, 0]))
    plt.bar(x - width, results_b[0][1].values[:, 1], width, color='cyan')
    plt.bar(x, results_b[1][1].values[:, 1], width, color='orange')
    plt.bar(x + width, results_b[2][1].values[:, 1], width, color='blue')
    plt.bar(x + width*2, results_b[3][1].values[:, 1], width, color='red')
    plt.xticks(x, results_b[0][1].values[:, 0])
    plt.legend(["user/user pearson", "user/user euclidean", "item/item pearson", "user / user filtered"])
    plt.title('Book-Crossing')
    plt.show()

    mask = [str(a) for a in np.arange(1., 11., 1)]
    x = np.arange(len(mask))
    plt.bar(x - width, results_b[0][2][mask].values[4], width, color='cyan')
    plt.bar(x, results_b[1][2][mask].values[4], width, color='orange')
    plt.bar(x + width, results_b[2][2][mask].values[4], width, color='blue')
    plt.bar(x + width*2, results_b[3][2][mask].values[4], width, color='red')
    plt.xticks(x, mask)
    plt.legend(["user/user pearson", "user/user euclidean", "item/item pearson", "user / user filtered"])
    plt.title('Book-Crossing - MSE distribution')
    plt.show()

    mask = [str(a) for a in np.arange(1., 11., 1)]
    x = np.arange(len(mask))
    plt.bar(x - width, results_b[0][2][mask].values[6], width, color='cyan')
    plt.bar(x, results_b[1][2][mask].values[6], width, color='orange')
    plt.bar(x + width, results_b[2][2][mask].values[6], width, color='blue')
    plt.bar(x + width*2, results_b[3][2][mask].values[6], width, color='red')
    plt.xticks(x, mask)
    plt.legend(["user/user pearson", "user/user euclidean", "item/item pearson", "user / user filtered"])
    plt.title('Book-Crossing - MAE distribution')
    plt.show()

    thr = 8
    mask = results_b[0][3].loc[0].values[1:] >= thr
    vals = results_b[0][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4u = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_b[1][3].loc[0].values[1:] >= thr
    vals = results_b[1][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4ue = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_b[2][3].loc[0].values[1:] >= thr
    vals = results_b[2][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4i = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()
    mask = results_b[3][3].loc[0].values[1:] >= thr
    vals = results_b[3][3].values[1:, 1:]
    vals = vals[:, mask]
    mse_4t = (vals[0, :] * vals[3, :]).sum() / vals[0, :].sum()

    x = np.arange(1)
    plt.bar(x - width, mse_4u, width, color='cyan')
    plt.bar(x, mse_4ue, width, color='orange')
    plt.bar(x + width, mse_4i, width, color='blue')
    plt.bar(x + width*2, mse_4t, width, color='red')
    plt.xticks(x, [''])
    plt.ylabel('average mse')
    plt.legend(["user/user pearson", "user/user euclidean", "item/item pearson", "nonzero"])
    plt.title('Book-Crossing - mse for predictions >' + str(thr))
    plt.show()

    print()


def load_results_movie():
    result_user = []
    result_ncf = []
    result_user_eu = []
    result_item = []
    for i in range(4):
        result_user.append(pd.read_csv('perf/person_user_' + str(i) + '.csv'))
        result_ncf.append(pd.read_csv('perf/ncf_mov_' + str(i) + '.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_basic.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_errors_total.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_group_actual.csv'))
    result_user_eu.append(pd.read_csv('perf/0distance_movie_group_predicted.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_basic.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_errors_total.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_group_actual.csv'))
    result_item.append(pd.read_csv('perf/0item_movie_group_predicted.csv'))
    return result_user, result_user_eu, result_ncf, result_item


def load_results_book():
    result_user = []
    result_user_eu = []
    result_item = []
    result_test = []
    for i in range(4):
        result_user.append(pd.read_csv('perf/person_user_book_' + str(i) + '.csv'))
        result_test.append(pd.read_csv('perf/person_user_test_' + str(i) + '.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_basic.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_errors_total.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_group_actual.csv'))
    result_user_eu.append(pd.read_csv('perf/5distance_book_group_predicted.csv'))
    result_item.append(pd.read_csv('perf/0item_book_basic.csv'))
    result_item.append(pd.read_csv('perf/0item_book_errors_total.csv'))
    result_item.append(pd.read_csv('perf/0item_book_group_actual.csv'))
    result_item.append(pd.read_csv('perf/0item_book_group_predicted.csv'))
    return result_user, result_user_eu, result_item, result_test


if __name__ == "__main__":
    main()
