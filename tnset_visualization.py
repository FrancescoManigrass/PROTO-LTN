import matplotlib
import numpy as np
import pandas
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

base_path = "ProtoLTN\\AwA2\\"
dataframe=pandas.read_csv("training_results\\PROT-545\\prototypes.csv").to_numpy()
prototype_df=dataframe[:,1:]


att_splits = scipy.io.loadmat(base_path + "att_splits.mat") #FOR AWA2
classes_names = att_splits['allclasses_names']
classes_names = [classes_names[i][0][0] for i in range(classes_names.size)]
for f in range(len(classes_names)):
    if "+" in classes_names[f]:
        v=classes_names[f].split("+")
        tmp=v[0]+"+"+v[1][0]+"."
        classes_names[f] = tmp

pca = PCA(n_components=50)
X_pca = pca.fit_transform(prototype_df)

tsne = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(X_pca)
df_subset = {}
df_subset['tsne-2d-one'] = tsne[:, 0]
df_subset['tsne-2d-two'] = tsne[:, 1]
df_subset['class'] = classes_names#[f.replace("+","\n") for f in classes_names]

df_subset['tsne-2d-one'] = (df_subset['tsne-2d-one'] - np.min(df_subset['tsne-2d-one'])) / (np.max(df_subset['tsne-2d-one']) - np.min(df_subset['tsne-2d-one']))
df_subset['tsne-2d-two'] = (df_subset['tsne-2d-two'] - np.min(df_subset['tsne-2d-two'])) / (np.max(df_subset['tsne-2d-two']) - np.min(df_subset['tsne-2d-two']))

plt.figure(figsize=(14, 10))
sns.set_context('paper', font_scale=2.5)
ax=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="class",
    palette=sns.color_palette("hls", 50),
    data=df_subset,
    legend=False,
    alpha=1, s=300
)

#plt.title('Awa2 Attribute Prototypes')
ax.set(xlabel=None)
ax.set(ylabel=None)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
size=50



def label_point(x, y, val):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    added=[]
    for i, point in a.iterrows():

        x= point['x']
        y= point['y']
        for j in added:
            if abs(j[0] - x)<=0.05 and  abs(j[1]-y)<=0.05 :

                x-=0.05

                y-= 0.05


        added.append((x, y))

        plt.text(x, y, str(point['val']),horizontalalignment='center')

label_point(pd.Series(list(df_subset['tsne-2d-one'])),pd.Series(list(df_subset['tsne-2d-two'])), pd.Series(df_subset['class']))

# plt.figure(dpi=400)

plt.savefig("feature_tsnet_" + "features.png")
plt.savefig("feature_tsnet_" + "features.pdf", bbox_inches='tight')
plt.show()
print("saved")

print("dfdfd")