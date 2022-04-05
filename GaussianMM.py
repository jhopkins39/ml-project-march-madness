import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import csv
import dataset

def process_data():
    # don't quite know what the parameters are in this method - will need to change
    data = pd.read_csv('data/RegularSeasonDetailedResults.csv', skiprows=1, index_col='date')
    data = np.array(data)

    teamID_data = pd.read_csv('data/Teams.csv', skiprows=1, index_col='date')
    # turn this into a dictionary to easily get the team names


# method body taken from the scikit-learn website
def use_model():

    ds = dataset.Dataset()
    curr_data = ds.getRegularGames()
    curr_data = np.array(curr_data)

    X_train = curr_data

    print(X_train)


    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20.0, 30.0)
    y = np.linspace(-20.0, 40.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)

    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.show()