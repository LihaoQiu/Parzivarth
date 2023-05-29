# import pandas as pd
# import glob
# import numpy as np
# import math
# from sklearn.decomposition import PCA, FastICA
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
#
#
# # Get the iris dataset
# dis_sum = []
# file_sum = []
#
#
# def shortest_distance(x1, y1, a, b, c):
#     d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
#     return d
#
#
# files = glob.glob('*.csv')
#
#
# for dataframe in files:
#     df = pd.read_csv(dataframe, header=None, index_col=False)
#     file_sum.append(dataframe)
#     all_points = df.iloc[:, 0:3]
#     label = df.iloc[:, 3]
#     # create figure
#     rails = pd.DataFrame()
#     rails_pca = pd.DataFrame()
#     rails_ica = pd.DataFrame()
#     rail1_pca = pd.DataFrame()
#     rail2_pca = pd.DataFrame()
#     rail1_ica = pd.DataFrame()
#     rail2_ica = pd.DataFrame()
#     rails_pi = pd.DataFrame()
#     rail1_project = pd.DataFrame()
#     rail2_project = pd.DataFrame()
#     for i in range(len(df)):
#         if df.iloc[i, 3] == 1:
#             data = df.iloc[i]
#             # data = data.values.reshape(1, 4)
#             # data = pd.Series(data)
#             rails = rails.append(data)
#     rail_plot = rails.iloc[:, 0:3]
#     my_dpi = 12
#     # plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)
#
#     # Run The PCA
#     pca = PCA(n_components=3)
#     pca.fit(all_points)
#
#     # ica = FastICA(n_components=3, random_state=0)
#     # ica.fit(all_points)
#
#     # Store results of PCA in a data frame
#     result_pca = pd.DataFrame(pca.transform(all_points), columns=['PCA%i' % i for i in range(3)], index=all_points.index)
#     # result_ica = pd.DataFrame(ica.transform(all_points), columns=['PCA%i' % i for i in range(3)], index=all_points.index)
#     # ica.fit(result_pca)
#     # result_pi = pd.DataFrame(ica.transform(result_pca), columns=['PCA%i' % i for i in range(3)], index=result_pca.index)
#     # First PCA
#     for i in range(len(result_pca)):
#         if label.iloc[i] == 1:
#             data = result_pca.iloc[i]
#             rails_pca = rails_pca.append(data)  # PCA railroad points
#
#     rails_project = rails_pca[['PCA0', 'PCA1']]
#
#     # for i in range(len(rails_project)):
#     #     rails_project.iloc[i, 2] = 0
#
#     # for i in range(len(result_pi)):
#     #     if label.iloc[i] == 1:
#     #         data = result_pi.iloc[i]
#     #         rails_pi = rails_pi.append(data)  # PCA railroad points
#     #
#     # for i in range(len(result_ica)):
#     #     if label.iloc[i] == 1:
#     #         data = result_ica.iloc[i]
#     #         rails_ica = rails_ica.append(data)  # PCA railroad points
#
#     for i in range(len(rails_pca)):
#         if rails_pca.iloc[i, 0] < 0:
#             a = rails_pca.iloc[i]
#             rail1_pca = rail1_pca.append(a)
#         else:
#             b = rails_pca.iloc[i]
#             rail2_pca = rail2_pca.append(b)  # PCA railroad points
#     #
#     # for i in range(len(rails_ica)):
#     #     if rails_ica.iloc[i, 0] < 0:
#     #         a = rails_ica.iloc[i]
#     #         rails_ica = rail1_pca.append(a)
#     #     else:
#     #         b = rails_pca.iloc[i]
#     #         rail2_pca = rail2_pca.append(b)  # PCA railroad points
#
#
#     # Plot initialisation
#     # fig = plt.figure()
#     # rows = 2
#     # columns = 2
#     # ax = fig.add_subplot(rows, columns, 1)
#     # ax.scatter(rails_project['PCA0'], rails_project['PCA1'], c='r', cmap="Set2_r", s=60)
#
#     max1 = 0
#     max2 = 0
#
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(rails_project)
#     cluster = pd.DataFrame(kmeans.labels_.T)
#
#     for i in range(len(rails_project)):
#         if cluster.iloc[i, 0] == 0:
#             a = rails_project.iloc[i]
#             rail1_project = rail1_project.append(a)
#         else:
#             b = rails_project.iloc[i]
#             rail2_project = rail2_project.append(b)  # PCA railroad points
#     rail1_project.reset_index(drop=True, inplace=True)
#     rail2_project.reset_index(drop=True, inplace=True)
#     # ax = fig.add_subplot(rows, columns, 2)
#     # ax.scatter(rail1_project['PCA0'], rail1_project['PCA1'], c='r', cmap="Set2_r", s=60)
#     #
#     #
#     # ax = fig.add_subplot(rows, columns, 3)
#     # ax.scatter(rail2_project['PCA0'], rail2_project['PCA1'], c='r', cmap="Set2_r", s=60)
#     # image = ax.scatter(rail2_project['PCA0'], rail2_project['PCA1'], c='r', cmap="Set2_r", s=60)
#
#
#     reg1 = LinearRegression().fit(np.asarray(rail1_project['PCA0']).reshape(-1, 1), np.asarray(rail1_project['PCA1'])\
#                                   .reshape(-1, 1))
#     reg2 = LinearRegression().fit(np.asarray(rail2_project['PCA0']).reshape(-1, 1), np.asarray(rail2_project['PCA1'])\
#                                   .reshape(-1, 1))
#
#     for i in range(len(rail1_project)):
#         d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
#         if d > max1:
#             max1 = d
#
#     for i in range(len(rail2_project)):
#         d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
#         if d > max2:
#             max2 = d
#     index1 = []
#     index2 = []
#     for i in range(len(rail1_project)):
#         d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
#         if d > 0.8 * max1:
#             index1.append(i)
#
#     for i in range(len(rail2_project)):
#         d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
#         if d > 0.8 * max2:
#             index2.append(i)
#     rail1_project.drop(index1, inplace=True)
#     rail2_project.drop(index2, inplace=True)
#
#     reg3 = LinearRegression().fit(np.asarray(rail1_project['PCA0']).reshape(-1, 1), np.asarray(rail1_project['PCA1'])\
#                                   .reshape(-1, 1))
#     reg4 = LinearRegression().fit(np.asarray(rail2_project['PCA0']).reshape(-1, 1), np.asarray(rail2_project['PCA1'])\
#                                   .reshape(-1, 1))
#
#     y3 = reg3.predict(np.asarray(rail1_project['PCA0']).reshape(-1, 1))
#     y4 = reg4.predict(np.asarray(rail2_project['PCA0']).reshape(-1, 1))
#     y = np.concatenate((y3, y4), axis=0)
#
#     # ax = fig.add_subplot(rows, columns, 4)
#     # ax.scatter(rail1_project['PCA0'], rail1_project['PCA1'], c='r', cmap="Set2_r", s=60)
#     # ax.scatter(rail2_project['PCA0'], rail2_project['PCA1'], c='r', cmap="Set2_r", s=60)
#     # ax.scatter(rail1_project['PCA0'], y3, c='b', cmap="Set2_r", s=60)
#     # ax.scatter(rail2_project['PCA0'], y4, c='b', cmap="Set2_r", s=60)
#     k = (reg3.coef_ + reg4.coef_) / 2
#     distance = abs(reg3.intercept_ - reg4.intercept_) / math.sqrt(k**2 + 1)
#     dis_sum.append(distance)
#     # plt.show()
# dis_sum = np.asarray(dis_sum)
# file_sum = np.asarray(file_sum)


#%%
import pandas as pd
import glob
import numpy as np
import math
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
# from scipy.optimize import curve_fit
# from sklearn.svm import SVR
# from scipy.optimize import least_squares
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import open3d as o3d

content = 'General'  # 'General', 'Gauge', 'Height'

# def func(x, a, b, c):
#     return a*np.sqrt(x)*(b*np.square(x)+c)

def func(x, a, b, c):
    return a * (x ** 2) + b * x + c

def linearfunc(x, a, b):
    return a * x + b


def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d


dis_sum = []
rail1_height_sum = []
rail2_height_sum = []
dist_var_sum = []
rail1_var_sum = []
rail2_var_sum = []
dis_median = []
files = glob.glob('/home/lihaoq/Documents/RandLA-Net-master/20220221/90/*.csv')
inner_distance = []
rail1_height = []
rail2_height = []
End1 = []
End2 = []
Mid = []

for file in files:

    df = pd.read_csv(file, header=None, index_col=False)
    all_points = df.iloc[:, 0:3]
    label = df.iloc[:, 3]
    # create figure
    rails = pd.DataFrame()
    rails_pca = pd.DataFrame()
    rails_pca2 = pd.DataFrame()
    rail1_pca = pd.DataFrame()
    rail2_pca = pd.DataFrame()
    rail1_project = pd.DataFrame()
    rail2_project = pd.DataFrame()
    for i in range(len(df)):
        if df.iloc[i, 3] == 1:
            data = df.iloc[i]
            rails = rails.append(data)
    rail_plot = rails.iloc[:, 0:3]
    my_dpi = 1
    # plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)


    # Run The PCA
    pca = PCA(n_components=3)
    pca.fit(all_points)

    # Store results of PCA in a data frame
    result_pca = pd.DataFrame(pca.transform(all_points), columns=['PCA%i' % i for i in range(3)], index=all_points.index)
    result_height = result_pca['PCA2']
    # plt.hist(result_height, bins='auto')
    # plt.show()
    # First PCA
    for i in range(len(result_pca)):
        if label.iloc[i] == 1:
            data = result_pca.iloc[i]
            rails_pca = rails_pca.append(data)  # PCA railroad points

    pca.fit(rails_pca)
    rails_pca2 = pd.DataFrame(pca.transform(rails_pca), columns=['PCA%i' % i for i in range(3)], index=rails_pca.index)

    rails_project = rails_pca[['PCA0', 'PCA1']]
    rails_height = rails_pca['PCA2']
    rows = 3
    columns = 3
    max1 = 0
    max2 = 0
    kmeans = KMeans(n_clusters=2, random_state=0).fit(rails_pca2)
    cluster = pd.DataFrame(kmeans.labels_.T)

    for i in range(len(rails_pca2)):  # Or rails_pca
        if cluster.iloc[i, 0] == 0:
            a = rails_project.iloc[i]
            c = rails_pca2.iloc[i]
            rail1_project = rail1_project.append(a)
            rail1_pca = rail1_pca.append(c)
        else:
            b = rails_project.iloc[i]
            d = rails_pca2.iloc[i]
            rail2_project = rail2_project.append(b)  # PCA railroad points
            rail2_pca = rail2_pca.append(d)
    rail1_project.reset_index(drop=True, inplace=True)
    rail2_project.reset_index(drop=True, inplace=True)
    rail1_pca.reset_index(drop=True, inplace=True)
    rail2_pca.reset_index(drop=True, inplace=True)

    axis_list = pca.components_
    x_axis, y_axis, z_axis = axis_list

    # row = 1
    # column = 2
    # fig = plt.figure(figsize=(60, 30))
    # ax = fig.add_subplot(row, column, 1)
    # ax.scatter(rail1_project['PCA0'], rail1_project['PCA1'], s=10)
    # ax.scatter(rail2_project['PCA0'], rail2_project['PCA1'], s=10)

    reg1 = LinearRegression().fit(np.asarray(rail1_project['PCA0']).reshape(-1, 1), np.asarray(rail1_project['PCA1']) \
                                  .reshape(-1, 1))
    reg2 = LinearRegression().fit(np.asarray(rail2_project['PCA0']).reshape(-1, 1), np.asarray(rail2_project['PCA1']) \
                                  .reshape(-1, 1))
    for i in range(len(rail1_project)):
        d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
        if d > max1:
            max1 = d

    for i in range(len(rail2_project)):
        d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
        if d > max2:
            max2 = d
    index1 = []
    index2 = []

    # LR1, removing outliers
    for i in range(len(rail1_project)):
        d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
        if d > 0.8 * max1:
            index1.append(i)

    for i in range(len(rail2_project)):
        d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
        if d > 0.8 * max2:
            index2.append(i)
    rail1_project.drop(index1, inplace=True)
    rail2_project.drop(index2, inplace=True)
    rail1_pca.drop(index1, inplace=True)
    rail2_pca.drop(index2, inplace=True)
    rail1_project.reset_index(drop=True, inplace=True)
    rail2_project.reset_index(drop=True, inplace=True)
    rail1_pca.reset_index(drop=True, inplace=True)
    rail2_pca.reset_index(drop=True, inplace=True)

    # ax = fig.add_subplot(row, column, 2)
    # ax.scatter(rail1_project['PCA0'], rail1_project['PCA1'], s=10)
    # ax.scatter(rail2_project['PCA0'], rail2_project['PCA1'], s=10)
    # plt.show()


    if content == 'Gauge':
        rail1_height = np.amax(np.asarray(rail1_pca['PCA2'])) - np.amin(np.asarray(rail1_pca['PCA2']))
        rail2_height = np.amax(np.asarray(rail2_pca['PCA2'])) - np.amin(np.asarray(rail2_pca['PCA2']))
        rail1_height_sum.append(rail1_height)
        rail2_height_sum.append(rail2_height)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(rows, columns, 1, projection='3d')
        ax.scatter(all_points.iloc[:, 0], all_points.iloc[:, 1], all_points.iloc[:, 2],c='r', cmap="Set2_r", s=1)
        ax = fig.add_subplot(rows, columns, 2, projection='3d')
        ax.scatter(result_pca['PCA0'], result_pca['PCA1'], result_pca['PCA2'], c='r', cmap="Set2_r", s=1)
        ax = fig.add_subplot(rows, columns, 3)
        ax.scatter(rails_project['PCA0'], rails_project['PCA1'], c='r', cmap="Set2_r", s=1)


        ax = fig.add_subplot(rows, columns, 4)
        ax.scatter(rail1_project['PCA0'], rail1_project['PCA1'], c='r', cmap="Set2_r", s=1)

        ax = fig.add_subplot(rows, columns, 5)
        ax.scatter(rail2_project['PCA0'], rail2_project['PCA1'], c='r', cmap="Set2_r", s=1)


        reg1 = LinearRegression().fit(np.asarray(rail1_project['PCA0']).reshape(-1, 1), np.asarray(rail1_project['PCA1'])\
                                      .reshape(-1, 1))
        reg2 = LinearRegression().fit(np.asarray(rail2_project['PCA0']).reshape(-1, 1), np.asarray(rail2_project['PCA1'])\
                                      .reshape(-1, 1))

        for i in range(len(rail1_project)):
            d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
            if d > max1:
                max1 = d

        for i in range(len(rail2_project)):
            d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
            if d > max2:
                max2 = d
        index1 = []
        index2 = []
        for i in range(len(rail1_project)):
            d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
            if d > 0.8 * max1:
                index1.append(i)

        for i in range(len(rail2_project)):
            d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
            if d > 0.8 * max2:
                index2.append(i)
        rail1_project.drop(index1, inplace=True)
        rail2_project.drop(index2, inplace=True)
        rail1_pca.drop(index1, inplace=True)
        rail2_pca.drop(index2, inplace=True)

        # Plotting
        # ax = fig.add_subplot(rows, columns, 6)
        # ax.scatter(rail1_project['PCA0'], rail1_project['PCA1'], c='r', cmap="Set2_r", s=1)
        #
        # ax = fig.add_subplot(rows, columns, 7)
        # ax.scatter(rail2_project['PCA0'], rail2_project['PCA1'], c='r', cmap="Set2_r", s=1)


        # plt.figure(1)
        # plt.hist(rail1_pca['PCA2'], bins='auto')
        # plt.figure(2)
        # plt.hist(rail2_pca['PCA2'], bins='auto')
        # plt.show()
        reg3 = LinearRegression().fit(np.asarray(rail1_project['PCA0']).reshape(-1, 1), np.asarray(rail1_project['PCA1'])\
                                      .reshape(-1, 1))
        reg4 = LinearRegression().fit(np.asarray(rail2_project['PCA0']).reshape(-1, 1), np.asarray(rail2_project['PCA1'])\
                                      .reshape(-1, 1))

        y3 = reg3.predict(np.asarray(rail1_project['PCA0']).reshape(-1, 1))
        y4 = reg4.predict(np.asarray(rail2_project['PCA0']).reshape(-1, 1))
        y = np.concatenate((y3, y4), axis=0)

        xyz = np.asarray(rail1_pca)
        rail1_pcd = o3d.open3d.geometry.PointCloud()
        rail1_pcd.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail1_pca))

        rail2_pcd = o3d.open3d.geometry.PointCloud()
        rail2_pcd.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail2_pca))
        dist = o3d.open3d.geometry.PointCloud.compute_point_cloud_distance(rail1_pcd, rail2_pcd)
        dist = np.asarray(dist)
        print('Mean rail distance', np.mean(dist), 'Rail distance median', np.median(dist), 'Rail distance var', np.var(dist))
        print('Rail1 height', np.mean(rail1_height), 'Rail1 height var', np.var(rail1_height))
        print('Rail2 height', np.mean(rail2_height), 'Rail2 height var', np.var(rail2_height))
        dist_var_sum.append(np.var(dist))
        rail1_var_sum.append(np.var(rail1_height))
        rail2_var_sum.append(np.var(rail2_height))

        print('input')

        N = 2000
        k = (reg3.coef_ + reg4.coef_) / 2
        distance = abs(reg3.intercept_ - reg4.intercept_) / math.sqrt(k**2 + 1)

        dis_sum.append(np.mean(dist))
        dis_median.append(np.median(dist))
        # plt.show()

    elif content == 'Height Calculation':
        reg1 = LinearRegression().fit(np.asarray(rail1_project['PCA0']).reshape(-1, 1), np.asarray(rail1_project['PCA1'])\
                                      .reshape(-1, 1))
        reg2 = LinearRegression().fit(np.asarray(rail2_project['PCA0']).reshape(-1, 1), np.asarray(rail2_project['PCA1'])\
                                      .reshape(-1, 1))
        for i in range(len(rail1_project)):
            d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
            if d > max1:
                max1 = d

        for i in range(len(rail2_project)):
            d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
            if d > max2:
                max2 = d
        index1 = []
        index2 = []
        for i in range(len(rail1_project)):
            d = shortest_distance(rail1_project.iloc[i, 0], rail1_project.iloc[i, 1], reg1.coef_, -1, reg1.intercept_)
            if d > 0.8 * max1:
                index1.append(i)

        for i in range(len(rail2_project)):
            d = shortest_distance(rail2_project.iloc[i, 0], rail2_project.iloc[i, 1], reg2.coef_, -1, reg2.intercept_)
            if d > 0.8 * max2:
                index2.append(i)
        rail1_project.drop(index1, inplace=True)
        rail2_project.drop(index2, inplace=True)
        rail1_pca.drop(index1, inplace=True)
        rail2_pca.drop(index2, inplace=True)
        rail1_project.reset_index(drop=True, inplace=True)
        rail2_project.reset_index(drop=True, inplace=True)
        rail1_pca.reset_index(drop=True, inplace=True)
        rail2_pca.reset_index(drop=True, inplace=True)
        pca.fit(rail1_pca)
        # Store results of PCA in a data frame
        rail1_pca2 = pd.DataFrame(pca.transform(rail1_pca), columns=['PCA%i' % i for i in range(3)],
                                  index=rail1_pca.index)
        pca.fit(rail2_pca)
        # Store results of PCA in a data frame
        rail2_pca2 = pd.DataFrame(pca.transform(rail2_pca), columns=['PCA%i' % i for i in range(3)],
                                  index=rail2_pca.index)

        index1t = []
        index1b = []
        index2t = []
        index2b = []
        rail1max = np.amax(np.asarray(rail1_pca['PCA2']))
        rail2max = np.amax(np.asarray(rail2_pca['PCA2']))
        rail1min = np.amin(np.asarray(rail1_pca['PCA2']))
        rail2min = np.amin(np.asarray(rail2_pca['PCA2']))
        # for i in range(len(rail1_pca)):
        #     if rail1_pca['PCA2'].iloc[i] > 0.05 * rail1max:
        #         index1t.append(i)
        #     if rail1_pca['PCA2'].iloc[i] < 0.5 * rail1min:
        #         index1b.append(i)
        # for i in range(len(rail2_pca)):
        #     if rail2_pca['PCA2'].iloc[i] > 0.05 * rail2max:
        #         index2t.append(i)
        #     if rail2_pca['PCA2'].iloc[i] < 0.5 * rail2min:
        #         index2b.append(i)
        # index1t = np.asarray(index1t)
        # index1b = np.asarray(index1b)
        # index2t = np.asarray(index2t)
        # index2b = np.asarray(index2b)
        # rail1_top = rail1_pca.loc[index1t]
        # rail1_bot = rail1_pca.loc[index1b]
        # rail2_top = rail1_pca.loc[index2t]
        # rail2_bot = rail1_pca.loc[index2b]
        # rail1_top.reset_index(drop=True, inplace=True)
        # rail1_bot.reset_index(drop=True, inplace=True)
        # rail2_top.reset_index(drop=True, inplace=True)
        # rail2_bot.reset_index(drop=True, inplace=True)
        # reg5 = LinearRegression().fit(np.asarray(rail1_top[['PCA0', 'PCA1']]).reshape(-1, 2), np.asarray(rail1_top['PCA2']))
        # y5 = reg5.predict(np.asarray(rail1_top[['PCA0', 'PCA1']]).reshape(-1, 2))
        # reg6 = LinearRegression().fit(np.asarray(rail1_bot[['PCA0', 'PCA1']]).reshape(-1, 2), np.asarray(rail1_bot['PCA2']))
        # y6 = reg6.predict(np.asarray(rail1_bot[['PCA0', 'PCA1']]).reshape(-1, 2))
        # rail1_top['PCA2'] = y5
        # rail1_bot['PCA2'] = y6
        # rail1_top_open = o3d.open3d.geometry.PointCloud()
        # rail1_top_open.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail1_top))
        #
        # rail1_bot_open = o3d.open3d.geometry.PointCloud()
        # rail1_bot_open.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail1_bot))
        # height1 = o3d.open3d.geometry.PointCloud.compute_point_cloud_distance(rail1_top_open, rail1_bot_open)
        # height1 = np.asarray(height1)
        # print('Height1_OPEN=', np.mean(height1))
        print('Height1=', rail1max-rail1min)
        # reg7 = LinearRegression().fit(np.asarray(rail2_top[['PCA0', 'PCA1']]).reshape(-1, 2), np.asarray(rail2_top['PCA2']))
        # y7 = reg7.predict(np.asarray(rail2_top[['PCA0', 'PCA1']]).reshape(-1, 2))
        # reg8 = LinearRegression().fit(np.asarray(rail2_bot[['PCA0', 'PCA1']]).reshape(-1, 2), np.asarray(rail2_bot['PCA2']))
        # y8 = reg8.predict(np.asarray(rail2_bot[['PCA0', 'PCA1']]).reshape(-1, 2))
        # rail2_top['PCA2'] = y7
        # rail2_bot['PCA2'] = y8
        # rail2_top_open = o3d.open3d.geometry.PointCloud()
        # rail2_top_open.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail2_top))
        #
        # rail2_bot_open = o3d.open3d.geometry.PointCloud()
        # rail2_bot_open.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail2_bot))
        # height2 = o3d.open3d.geometry.PointCloud.compute_point_cloud_distance(rail2_top_open, rail2_bot_open)
        # height2 = np.asarray(height2)
        # print('Height2_OPEN=', np.mean(height2))
        print('Height2=', rail2max-rail2min)

    elif content == 'General':
        rail1_pca_height = rail1_pca.sort_values(by='PCA2', ascending=False).reset_index(drop=True)
        rail2_pca_height = rail2_pca.sort_values(by='PCA2', ascending=False).reset_index(drop=True)

        reg3 = LinearRegression().fit(np.asarray(rail1_pca_height['PCA0']).reshape(-1, 1), np.asarray(rail1_pca_height['PCA1']) \
                                      .reshape(-1, 1))
        reg4 = LinearRegression().fit(np.asarray(rail1_pca_height['PCA0']).reshape(-1, 1), np.asarray(rail1_pca_height['PCA2']) \
                                      .reshape(-1, 1))
        reg5 = LinearRegression().fit(np.asarray(rail2_pca_height['PCA0']).reshape(-1, 1), np.asarray(rail2_pca_height['PCA1']) \
                                      .reshape(-1, 1))
        reg6 = LinearRegression().fit(np.asarray(rail2_pca_height['PCA0']).reshape(-1, 1), np.asarray(rail2_pca_height['PCA2']) \
                                      .reshape(-1, 1))

        # Calculate Warp
        reg7 = LinearRegression().fit(np.asarray(rail1_pca_height['PCA1']).reshape(-1, 1), np.asarray(rail1_pca_height['PCA2']) \
                                      .reshape(-1, 1))
        reg8 = LinearRegression().fit(np.asarray(rail2_pca_height['PCA1']).reshape(-1, 1), np.asarray(rail2_pca_height['PCA2']) \
                                      .reshape(-1, 1))

        ra1min = rail1_pca_height['PCA1'].min()
        ra1max = rail1_pca_height['PCA1'].max()
        ra2min = rail2_pca_height['PCA1'].min()
        ra2max = rail2_pca_height['PCA1'].max()

        if ra1min > ra2min:
            left = ra1min
        else:
            left = ra2min

        if ra1max > ra2max:
            right = ra2max
        else:
            right = ra1max

        ra1int = (ra1max - ra1min) / 5
        ra2int = (ra2max - ra2min) / 5

        # Print warp data

        warp = []
        legend = []
        warp_append = []
        for i in np.arange(left, right, 1/len(rail1_pca_height)):
            dif = (reg7.coef_ * i + reg7.intercept_) - (reg8.coef_ * i + reg8.intercept_)
            warp.append(dif)
        warp = np.array(warp)
        warp = warp.squeeze()

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(rail1_pca_height['PCA1'], reg7.coef_*np.asarray(rail1_pca_height['PCA1']) + reg7.intercept_, s=1)
        ax.scatter(rail2_pca_height['PCA1'], reg8.coef_*np.asarray(rail2_pca_height['PCA1']) + reg8.intercept_, s=1)
        # ax.scatter(np.arange(left, right, 1/len(rail1_pca_height)), warp, s=1)

        for i in range(6):
            wa = abs(reg7.coef_ * ra1min + reg7.intercept_ - (reg8.coef_ * (ra2min + i * ra2int) + reg8.intercept_))
            warp_append.append(wa)
            legend.append(wa)
            a1 = ra1min
            a2 = np.squeeze(reg7.coef_ * ra1min + reg7.intercept_)
            a3 = ra2min + i * ra2int - ra1min
            a4 = np.squeeze(reg8.coef_ * (ra2min + i * ra2int) + reg8.intercept_ - reg7.coef_ * ra1min - reg7.intercept_)
            plt.arrow(a1, a2, a3, a4, width=0.00001, head_length=0.03, head_width=0.00003)
        ax.legend(legend)
        plt.savefig(file + '11.png')
        plt.close()

        legend = []
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(rail1_pca_height['PCA1'], reg7.coef_*np.asarray(rail1_pca_height['PCA1']) + reg7.intercept_, s=1)
        ax.scatter(rail2_pca_height['PCA1'], reg8.coef_*np.asarray(rail2_pca_height['PCA1']) + reg8.intercept_, s=1)
        # ax.scatter(np.arange(left, right, 1/len(rail1_pca_height)), warp, s=1)

        for i in range(6):
            wa = abs(reg8.coef_ * ra2min + reg8.intercept_ - (reg7.coef_ * (ra1min + i * ra1int) + reg7.intercept_))
            warp_append.append(wa)
            legend.append(wa)
            a1 = ra2min
            a2 = np.squeeze(reg8.coef_ * ra2min + reg8.intercept_)
            a3 = ra1min + i * ra1int - ra2min
            a4 = np.squeeze(reg7.coef_ * (ra1min + i * ra1int) + reg7.intercept_ - reg8.coef_ * ra2min - reg8.intercept_)
            plt.arrow(a1, a2, a3, a4, width=0.00001, head_length=0.03, head_width=0.00003)
        ax.legend(legend)
        plt.savefig(file + '21.png')
        plt.close()


        # Print Gauge
        Max1end = 0
        Min1end = 0
        Max2end = 0
        Min2end = 0
        max3 = np.squeeze(rail1_pca_height.iloc[0, 0] + reg3.coef_ * np.asarray(rail1_pca_height.iloc[0, 0]) + reg3.intercept_)
        min3 = np.squeeze(rail1_pca_height.iloc[0, 0] + reg3.coef_ * np.asarray(rail1_pca_height.iloc[0, 0]) + reg3.intercept_)

        for i in range(2, len(rail1_pca_height)):
            if rail1_pca_height.iloc[i, 0] + reg3.coef_ * np.asarray(rail1_pca_height.iloc[i, 0]) + reg3.intercept_ > max3:
                Max1end = i
                max3 = rail1_pca_height.iloc[i, 0] + reg3.coef_ * np.asarray(rail1_pca_height.iloc[i, 0]) + reg3.intercept_
            if rail1_pca_height.iloc[i, 0] + reg3.coef_ * np.asarray(rail1_pca_height.iloc[i, 0]) + reg3.intercept_ < min3:
                Min1end = i
                min3 = rail1_pca_height.iloc[i, 0] + reg3.coef_ * np.asarray(rail1_pca_height.iloc[i, 0]) + reg3.intercept_

        max3 = rail2_pca_height.iloc[0, 0] + reg5.coef_ * np.asarray(rail2_pca_height.iloc[0, 0]) + reg5.intercept_
        minmin3 = rail2_pca_height.iloc[0, 0] + reg5.coef_ * np.asarray(rail2_pca_height.iloc[0, 0]) + reg5.intercept_

        for i in range(2, len(rail2_pca_height)):
            if rail2_pca_height.iloc[i, 0] + reg5.coef_ * np.asarray(rail2_pca_height.iloc[i, 0]) + reg5.intercept_ > max3:
                Max2end = i
                max3 = rail2_pca_height.iloc[i, 0] + reg5.coef_ * np.asarray(rail2_pca_height.iloc[i, 0]) + reg5.intercept_
            if rail2_pca_height.iloc[i, 0] + reg5.coef_ * np.asarray(rail2_pca_height.iloc[i, 0]) + reg5.intercept_ < min3:
                Min2end = i
                min3 = rail2_pca_height.iloc[i, 0] + reg5.coef_ * np.asarray(rail2_pca_height.iloc[i, 0]) + reg5.intercept_

        End1.append(rail1_pca_height.iloc[Max1end, 2] - rail2_pca_height.iloc[Max2end, 2])
        End2.append(rail1_pca_height.iloc[Min1end, 2] - rail2_pca_height.iloc[Min2end, 2])
        Mid.append((rail1_pca_height.iloc[Max1end, 2] - rail2_pca_height.iloc[Max2end, 2] + rail1_pca_height.iloc[Min1end, 2] - rail2_pca_height.iloc[Min2end, 2])/2)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(rail1_pca_height['PCA0'], reg3.coef_*np.asarray(rail1_pca_height['PCA0']) + reg3.intercept_,\
        #            reg4.coef_*np.asarray(rail1_pca_height['PCA0']) + reg4.intercept_, s=1)
        # ax.scatter(rail2_pca_height['PCA0'], reg5.coef_*np.asarray(rail2_pca_height['PCA0']) + reg5.intercept_,\
        #            reg6.coef_*np.asarray(rail2_pca_height['PCA0']) + reg6.intercept_, s=1)
        # ax.scatter(rail1_pca_height.iloc[Max1end, 0], reg3.coef_*np.asarray(rail1_pca_height.iloc[Max1end, 0]) + reg3.intercept_,\
        #            reg4.coef_*np.asarray(rail1_pca_height.iloc[Max1end, 0]) + reg4.intercept_, s=20)
        #
        # ax.scatter(rail1_pca_height.iloc[Min1end, 0], reg3.coef_*np.asarray(rail1_pca_height.iloc[Min1end, 0]) + reg3.intercept_,\
        #            reg4.coef_*np.asarray(rail1_pca_height.iloc[Min1end, 0]) + reg4.intercept_, s=20)
        #
        # ax.scatter(rail2_pca_height.iloc[Max2end, 0], reg5.coef_*np.asarray(rail2_pca_height.iloc[Max2end, 0]) + reg5.intercept_,\
        #            reg6.coef_*np.asarray(rail2_pca_height.iloc[Max2end, 0]) + reg6.intercept_, s=20)
        #
        # ax.scatter(rail2_pca_height.iloc[Min2end, 0], reg5.coef_*np.asarray(rail2_pca_height.iloc[Min2end, 0]) + reg5.intercept_,\
        #            reg6.coef_*np.asarray(rail2_pca_height.iloc[Min2end, 0]) + reg6.intercept_, s=20)
        # plt.xlabel('PCA0')
        # plt.ylabel('PCA1')
        # plt.show()

        rot_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1 - rail1_pca_height.iloc[Max1end, 2] + rail1_pca_height.iloc[Min1end, 2]]])
        rot_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1 - rail2_pca_height.iloc[Max2end, 2] + rail2_pca_height.iloc[Min2end, 2]]])

        rail1_pca_height = np.array(rail1_pca_height)
        rail2_pca_height = np.array(rail2_pca_height)


        rail1_pca_rot = np.dot(rail1_pca_height, rot_mat1)

        rail2_pca_rot = np.dot(rail2_pca_height, rot_mat2)

        # Plot histogram
        # fig = plt.figure(figsize=(60, 30))
        #
        # row = 1
        # column = 2
        # # ax = fig.add_subplot(row, column, 1, projection='3d')
        # # ax.scatter(rail1_pca_rot[:, 0], rail1_pca_rot[:, 1], rail1_pca_rot[:, 2], s=1)
        # # ax.scatter(rail2_pca_rot[:, 0], rail2_pca_rot[:, 1], rail2_pca_rot[:, 2], s=1)
        #
        # ax = fig.add_subplot(row, column, 1)
        # ax.set_title(file, fontsize=30)
        # ax.hist(rail1_pca_rot[:, 2], bins=20)
        #
        # ax = fig.add_subplot(row, column, 2)
        # ax.set_title(file, fontsize=30)
        # ax.hist(rail2_pca_rot[:, 2], bins=20)
        # plt.savefig(file + '.png')
        # plt.close()

        rail1_pca_height = pd.DataFrame(rail1_pca_height, columns=['PCA0', 'PCA1', 'PCA2'])
        rail2_pca_height = pd.DataFrame(rail2_pca_height, columns=['PCA0', 'PCA1', 'PCA2'])

        rail1_pca_rot = pd.DataFrame(rail1_pca_rot, columns=['PCA0', 'PCA1', 'PCA2'])
        rail2_pca_rot = pd.DataFrame(rail2_pca_rot, columns=['PCA0', 'PCA1', 'PCA2'])

        rail1_pca_rot = rail1_pca_rot.sort_values(by='PCA2', ascending=False).reset_index(drop=True)
        rail2_pca_rot = rail2_pca_rot.sort_values(by='PCA2', ascending=False).reset_index(drop=True)

        # fig = plt.figure(figsize=(60, 30))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(rail1_pca_rot['PCA0'], rail1_pca_rot['PCA1'], rail1_pca_rot['PCA2'], s=60)
        # ax.scatter(rail2_pca_rot['PCA0'], rail2_pca_rot['PCA1'], rail2_pca_rot['PCA2'], s=60)
        # plt.show()

        rail1_pca_top = rail1_pca_rot.iloc[int(0 * len(rail1_pca_rot)): int(0.03 * len(rail1_pca_rot)), :]
        rail2_pca_top = rail2_pca_rot.iloc[int(0 * len(rail2_pca_rot)): int(0.03 * len(rail2_pca_rot)), :]

        rail1_pca_bot = rail1_pca_rot.iloc[int(0.9 * len(rail1_pca_rot)): int(1 * len(rail1_pca_rot)), :]
        rail2_pca_bot = rail2_pca_rot.iloc[int(0.9 * len(rail2_pca_rot)): int(1 * len(rail2_pca_rot)), :]

        rail1_pca_cut = rail1_pca_rot.iloc[int(0.1 * len(rail1_pca_rot)): int(0.3 * len(rail1_pca_rot)), :]
        rail2_pca_cut = rail2_pca_rot.iloc[int(0.1 * len(rail2_pca_rot)): int(0.3 * len(rail2_pca_rot)), :]



        reg9 = LinearRegression().fit(np.asarray(rail1_pca_cut['PCA0']).reshape(-1, 1), np.asarray(rail1_pca_cut['PCA1']) \
                                      .reshape(-1, 1))
        reg10 = LinearRegression().fit(np.asarray(rail2_pca_cut['PCA0']).reshape(-1, 1), np.asarray(rail2_pca_cut['PCA1']) \
                                      .reshape(-1, 1))

        y3 = reg9.predict(np.asarray(rail1_pca_cut['PCA0']).reshape(-1, 1))
        y4 = reg10.predict(np.asarray(rail2_pca_cut['PCA0']).reshape(-1, 1))
        k = (reg9.coef_ + reg10.coef_) / 2
        distance = abs(reg9.intercept_ - reg10.intercept_) / math.sqrt(k**2 + 1)
        # print('Linear Regression Distance: ', distance)
        PCA3 = []
        for i in range(len(rail1_pca_cut)):
            dis = abs(reg9.coef_ * rail1_pca_cut.iloc[i, 0] - rail1_pca_cut.iloc[i, 1] + reg9.intercept_)\
                  / math.sqrt(reg9.coef_ ** 2 + 1)
            PCA3.append(dis)
        PCA3 = np.array(PCA3)
        PCA3 = np.squeeze(PCA3)
        rail1_pca_cut['PCA3'] = PCA3

        PCA3 = []  # Calculate distance to each other for filtering
        for i in range(len(rail2_pca_cut)):
            dis = abs(reg10.coef_ * rail2_pca_cut.iloc[i, 0] - rail2_pca_cut.iloc[i, 1] + reg10.intercept_)\
                  / math.sqrt(reg10.coef_ ** 2 + 1)
            PCA3.append(dis)
        PCA3 = np.array(PCA3)
        PCA3 = np.squeeze(PCA3)
        rail2_pca_cut['PCA3'] = PCA3

        rail1_pca_cut = rail1_pca_cut.sort_values(by='PCA3', ascending=True).reset_index(drop=True)
        rail2_pca_cut = rail2_pca_cut.sort_values(by='PCA3', ascending=True).reset_index(drop=True)

        rail1_pca_inner = rail1_pca_cut.iloc[int(0 * len(rail1_pca_cut)): int(0.1 * len(rail1_pca_cut)), :]
        rail2_pca_inner = rail2_pca_cut.iloc[int(0 * len(rail2_pca_cut)): int(0.1 * len(rail2_pca_cut)), :]

        rail1_pca_inner = rail1_pca_inner.iloc[:, 0:3]
        rail2_pca_inner = rail2_pca_inner.iloc[:, 0:3]

        reg11 = LinearRegression().fit(np.asarray(rail1_pca_inner['PCA0']).reshape(-1, 1), np.asarray(rail1_pca_inner['PCA1']) \
                                      .reshape(-1, 1))
        reg12 = LinearRegression().fit(np.asarray(rail2_pca_inner['PCA0']).reshape(-1, 1), np.asarray(rail2_pca_inner['PCA1']) \
                                      .reshape(-1, 1))

        y3 = reg11.predict(np.asarray(rail1_pca_inner['PCA0']).reshape(-1, 1))
        y4 = reg12.predict(np.asarray(rail2_pca_inner['PCA0']).reshape(-1, 1))
        k = (reg11.coef_ + reg12.coef_) / 2
        distance = abs(reg11.intercept_ - reg12.intercept_) / math.sqrt(k**2 + 1)
        inner_distance.append(distance)
        # print('Inner Linear Regression Distance: ', distance)   # Calculate Gauge

        # r1_top = np.argmax(np.asarray(rail1_pca['PCA2']))
        # r1_bot = np.argmin(np.asarray(rail1_pca['PCA2']))
        # r2_top = np.argmax(np.asarray(rail2_pca['PCA2']))
        # r2_bot = np.argmin(np.asarray(rail2_pca['PCA2']))
        # print('Rail1 top - bottom: ', rail1_pca_rot.iloc[0, 2] - rail1_pca_rot.iloc[len(rail1_pca_rot) - 1, 2])
        rail1_pca_rot = rail1_pca_rot.sort_values(by='PCA2', ascending=True).reset_index(drop=True)
        rail2_pca_rot = rail2_pca_rot.sort_values(by='PCA2', ascending=True).reset_index(drop=True)
        rail1_height.append(rail1_pca_rot.iloc[0, 2] - rail1_pca_rot.iloc[len(rail1_pca_rot) - 1, 2])
        # print('Rail2 top - bottom: ', rail2_pca_rot.iloc[0, 2] - rail2_pca_rot.iloc[len(rail2_pca_rot) - 1, 2])
        rail2_height.append(rail2_pca_rot.iloc[0, 2] - rail2_pca_rot.iloc[len(rail2_pca_rot) - 1, 2])
        #
        # rail1_pcd = o3d.open3d.geometry.PointCloud()
        # rail1_pcd.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail1_pca_inner))
        #
        # rail2_pcd = o3d.open3d.geometry.PointCloud()
        # rail2_pcd.points = o3d.open3d.utility.Vector3dVector(np.asarray(rail2_pca_inner))
        # dist = o3d.open3d.geometry.PointCloud.compute_point_cloud_distance(rail1_pcd, rail2_pcd)
        # dist = np.asarray(dist)
        # print(np.mean(dist))
        # fig = plt.figure()
        # row = 1
        # column = 1
        # ax = fig.gca(projection='3d')
        # ax.scatter(result_pca['PCA0'], result_pca['PCA1'], result_pca['PCA2'], c='r', s=1, label='polyfit', alpha=0.0)
        # plt.xlabel('PCA0')
        # plt.ylabel('PCA1')
        # ax.scatter(1 * X1, linear1, c='g', s=1, label='linear fit')
        # ax = fig.add_subplot(row, column, 2, projection='3d')
        #
        # ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], x_axis, y_axis, z_axis, lw=2, length=0.1)
        #
        # ax.scatter(rail1_pca_cut['PCA0'], rail1_pca_cut['PCA1'], rail1_pca_cut['PCA2'], s=1)
        # ax.scatter(rail2_pca_cut['PCA0'], rail2_pca_cut['PCA1'], rail2_pca_cut['PCA2'], s=1)

        # ax.scatter(rail1_pca_cut['PCA0'], rail1_pca_cut['PCA1'], rail1_pca_cut['PCA2'], c='r', s=1)
        # ax.scatter(rail2_pca_cut['PCA0'], rail2_pca_cut['PCA1'], rail2_pca_cut['PCA2'], c='r', s=1)
        # plt.xlabel('PCA0')
        # plt.ylabel('PCA1')
        # ax.scatter(1 * X2, rail2, c='b', s=1, label='polyfit')
        # # ax.scatter(1 * X2, linear2, c='y', s=1, label='linear fit')
        # ax.scatter(1 * X1, Y1, c='w', s=1, label='points', alpha=0)
        # ax.scatter(1 * X2, Y2, c='w', s=1, label='points', alpha=0)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend(loc=4)  # 指定legend的位置右下角
        # plt.title('curve_fit')
        # # plt.show()
        #
        # ax = fig.add_subplot(row, column, 2)
        # ax.scatter(1 * X1, rail1, c='r', s=1, label='polyfit')
        # ax.scatter(1 * X1, linear1, c='g', s=1, label='linear fit')
        # ax.scatter(1 * X1, Y1, c='b', s=1, label='points')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend(loc=4)  # 指定legend的位置右下角
        # plt.title('curve_fit')
        #
        # ax = fig.add_subplot(row, column, 3)
        # ax.scatter(1 * X2, rail2, c='r', s=1, label='polyfit')
        # ax.scatter(1 * X2, linear2, c='g', s=1, label='linear fit')
        # ax.scatter(1 * X2, Y2, c='b', s=1, label='points')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend(loc=4)  # 指定legend的位置右下角
        # plt.title('curve_fit')
        # plt.show()
    # dis_sum = np.asarray(dis_sum)
    # rail1_height_sum = np.asarray(rail1_height_sum)
    # rail2_height_sum = np.asarray(rail2_height_sum)
    # dist_var_sum = np.asarray(dist_var_sum)
    # dis_median = np.asarray(dis_median)
    # print('Rail 1 height mean',np.mean(rail1_height_sum), 'Rail 2 height mean', np.mean(rail2_height_sum),'Rail 1 height variance',\
    #       np.var(rail1_height_sum), 'Rail 2 height variance', np.var(rail2_height_sum))
    #     # file_sum = np.asarray(file_sum)
    # plt.figure(1)
    # plt.hist(rail1_height_sum)
    # plt.xlabel('value')
    # plt.ylabel('number')
    # plt.title('Rail1')
    #
    # plt.figure(2)
    # plt.hist(rail2_height_sum)
    # plt.xlabel('value')
    # plt.ylabel('number')
    # plt.title('Rail2')
    #
    # plt.show()
inner_distance = np.array(inner_distance)
dist_mean = np.mean(inner_distance)
dist_var = np.var(inner_distance)
rail1_height = np.array(rail1_height)
rail2_height = np.array(rail2_height)
# End1 = np.array(End1)
# End2 = np.array(End2)
# Mid = np.array(Mid)






