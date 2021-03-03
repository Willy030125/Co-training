from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV
from classifiers import CoTrainingClassifier
import os


if __name__ == '__main__':
	N_SAMPLES = 25000
	N_FEATURES = 1000
	X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)

	y[:N_SAMPLES//2] = -1

	X_test = X[-N_SAMPLES//4:]
	y_test = y[-N_SAMPLES//4:]

	X_labeled = X[N_SAMPLES//2:-N_SAMPLES//4]
	y_labeled = y[N_SAMPLES//2:-N_SAMPLES//4]

	y = y[:-N_SAMPLES//4]
	X = X[:-N_SAMPLES//4]


	X1 = X[:,:N_FEATURES // 2]
	X2 = X[:, N_FEATURES // 2:]


	f = open("result.txt", "w+")

	print('Logistic')
	base_lr = LogisticRegression()
	base_lr.fit(X_labeled, y_labeled)
	y_pred = base_lr.predict(X_test)
	lr_result = classification_report(y_test, y_pred)
	print(lr_result)
	f.write("Logistic")
	f.write("\n======================================================\n")
	f.write(str(lr_result))
	f.write("\n======================================================\n\n")

	print('Logistic CoTraining')
	lg_co_clf = CoTrainingClassifier(LogisticRegression())
	lg_co_clf.fit(X1, X2, y)
	y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	co_lr_result = classification_report(y_test, y_pred)
	print(co_lr_result)
	f.write("Logistic CoTraining")
	f.write("\n======================================================\n")
	f.write(str(co_lr_result))
	f.write("\n======================================================\n\n")

	print('SVM')
	base_svm = LinearSVC(dual=False)
	base_svm.fit(X_labeled, y_labeled)
	y_pred = base_svm.predict(X_test)
	svm_result = classification_report(y_test, y_pred)
	print(svm_result)
	f.write("SVM")
	f.write("\n======================================================\n")
	f.write(str(svm_result))
	f.write("\n======================================================\n\n")
	
	print('SVM CoTraining')
	svm = LinearSVC(dual=False)
	clf = CalibratedClassifierCV(svm)
	svm_co_clf = CoTrainingClassifier(clf, u=N_SAMPLES//10)
	svm_co_clf.fit(X1, X2, y)
	y_pred = svm_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	co_svm_result = classification_report(y_test, y_pred)
	print(co_svm_result)
	f.write("SVM CoTraining")
	f.write("\n======================================================\n")
	f.write(str(co_svm_result))
	f.write("\n======================================================\n\n")

	f.close()

	print("=== Process done, check result.txt file for the output. === \n")
	os.system("pause")