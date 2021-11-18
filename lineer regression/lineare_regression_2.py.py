# -*- coding: utf-8 -*-

# lineare_regression
#
# Routinen zur Berechnung der multivariaten linearen Regression mit Modell-
# funktion
#
#   h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
#
# und Kostenfunktion
#
#   J(theta) = 1/(2m) sum_i=1^m ( h_theta(x^(i)) - y^(i) )^2
#
# Der Vektor theta wird als
#
#   (theta_0, theta_1, ... , theta_n)
#
# gespeichert. Die Feature-Matrix mit m Daten und n Features als
#
#       [ - x^(1) - ]
#   X = [    .      ]    (m Zeilen und n Spalten)
#       [ - x^(m) - ]
#

import numpy as np
import pandas as pd

# %% extend_matrix

# Erweitert eine Matrix um eine erste Spalte mit Einsen
#
# X_ext = extend_matrix(X)
#
# Eingabe:
#   X      Matrix m x n (numpy.ndarray)
#
# Ausgabe
#   X_ext  Matrix m x (n+1) der Form [1 X] (numpy.ndarray)
#
def extend_matrix(X):
    # TODO: setze X_ext
    # X = X.transpose()

    X_ext = np.insert(X, 0, 1, axis=1)
    return X_ext


# %% LR_fit

# Berechnung der optimalen Parameter der multivariaten linearen Regression
# mithilfe der Normalengleichung.
#
# theta = LR_fit(X, y)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#
# Ausgabe
#   theta  Vektor der  Länge n+1 der optimalen Parameter (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix und np.linalg.solve zur Lösung des
#   linearen Gleichungssystems
#
def LR_fit(X, y):
    # TODO: berechne theta

    X_ext = extend_matrix(X)
    X_ext_trans = X_ext.transpose()
    a = np.matmul(X_ext_trans, X_ext)
    b = np.matmul(X_ext_trans, y)
    theta = np.linalg.solve(a, b)
    return theta


# %% LR_predict

# Berechnung der Vorhersage der der multivariaten linearen Regression.
#
# y = LR_predict(X,theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   y      Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix.
#
def LR_predict(X, theta):
    # TODO: berechne y
    X_ext = extend_matrix(X)
    y = np.matmul(X_ext, theta)
    return y


# %% r2_score

# Berechnung des Bestimmtheitsmaßes R2
#
# y = r2_score(X, y, theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   r2     Bestimmtheitsmaß R2 (Skalar)
#
# Hinweis: Benutzen Sie LR_predict
#
def r2_score(X, y, theta):
    # TODO: berechne r2

    h_theta = LR_predict(X,theta)
    y_mean = np.mean(y)
    c = np.sum((y-h_theta)**2)
    d = np.sum((y-y_mean)**2)
    r2 = 1-(c/d)
    return r2


if __name__ == '__main__':
    print(__name__)
    # Erzeuge Feature-Matrix & Zielvektor
    # X = np.array([[0, 6, 12]])
    # y = np.array([[13.5], [10.5], [3.0]])

    x1 = pd.read_csv("./multivariat.csv", sep=",", usecols=['x1'])
    x1 = x1.to_numpy()
    x2 = pd.read_csv("./multivariat.csv", sep=",", usecols=['x2'])
    x2 = x2.to_numpy()
    y = pd.read_csv("./multivariat.csv", sep=",", usecols=['y'])
    X = np.ones((np.shape(x1)[0], np.shape(x1)[1] + 1))
    X[:, 0] = x1[:, 0]
    X[:, 1] = x2[:, 0]
    y = y.to_numpy()

    # print("Das ist die Featurematrix X: ")
    # print(X)
    # print("Das ist der Zielvektor y: ")
    # print(y)

    print(r2_score(X,y,LR_fit(X,y)))

