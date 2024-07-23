from cmath import inf
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.svm import SVC

#Exercise 2
#Usage: python3 ex2.py
#Load the files in some other way (e.g., pandas) if you prefer
X_train_A = np.loadtxt('data/X_train_A.csv', delimiter=",")
Y_train_A = np.loadtxt('data/Y_train_A.csv',  delimiter=",").astype(int)

X_train_B = np.loadtxt('data/X_train_B.csv', delimiter=",")
Y_train_B = np.loadtxt('data/Y_train_B.csv', delimiter=",").astype(int)
X_test_B = np.loadtxt('data/X_test_B.csv', delimiter=",")
Y_test_B = np.loadtxt('data/Y_test_B.csv', delimiter=",").astype(int)


# Part 2.1
try:
    log_reg = sm.Logit(Y_train_A, X_train_A).fit()
except:
    print("logistic regression fails due to perfect separation")

# SVM approach
sm_svm = SVC(C=1, kernel="linear")
hm_svm = SVC(C=float('inf'), kernel="linear")
sm_svm.fit(X_train_A, Y_train_A)
hm_svm.fit(X_train_A, Y_train_A)

sm_w = sm_svm.coef_[0]
sm_intercept = sm_svm.intercept_
hm_w = hm_svm.coef_[0]
hm_intercept = hm_svm.intercept_
print(f"soft margin weights = {sm_w}")
print(f"soft margin bias = {sm_intercept}")
print(f"hard margin weights = {hm_w}")
print(f"soft margin bias = {hm_intercept}")


# Part 2.2

count = 0

for x_i, y_i in zip(X_train_A, Y_train_A):
    dot_prod = np.dot(sm_w, x_i)
    print(f"Dot_prod = {dot_prod}")
    sign = 1 if y_i == 1 else -1
    if dot_prod * sign <= 1:
        count += 1
print(f"Part 2.2: Count = {count}")
print(f"Number of total points in X_Train_A = {len(X_train_A)}")

