
from tkinter import *
import os
# from gui_stuff import *
from PIL import ImageTk,Image
from tkinter import messagebox



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize lists to store model results
acc = []
model = []

# Load data
df = pd.read_csv('data/cp1.csv')
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Split data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Linear SVM class implementation
class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        y = np.array(y)
        X = self.scaler.fit_transform(X)

        # Encode labels for binary classification
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        X = self.scaler.transform(X)  # Apply scaling to test data
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx).astype(int)

def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_num[label] for label in labels])
    return encoded_labels, label_to_num

# Function to execute Linear SVM prediction and display results
def func_LinearSVM():
    # Encode target labels as integers manually
    Ytrain_encoded, label_to_num = encode_labels(Ytrain)
    Ytest_encoded, _ = encode_labels(Ytest)

    # Train the model
    linear_svm = LinearSVM()
    linear_svm.fit(Xtrain.values, Ytrain_encoded)

    # Predict
    predicted_values = linear_svm.predict(Xtest.values)

    # Compute accuracy
    accuracy = np.mean(predicted_values == Ytest_encoded)
    acc.append(accuracy)
    model.append('Linear SVM')

    # Display predictions
    try:
        N = float(nty_N.get())
        P = float(nty_P.get())
        K = float(nty_K.get())
        Temperature = float(nty_T.get())
        Humidity = float(nty_H.get())
        ph = float(nty_Ph.get())
        Rainfall = float(nty_R.get())

        data = np.array([[N, P, K, Temperature, Humidity, ph, Rainfall]])
        prediction = linear_svm.predict(data)

        # Decode prediction
        predicted_label = [list(label_to_num.keys())[list(label_to_num.values()).index(pred)] for pred in prediction]

        Pdt_svm = Label(root, text=predicted_label[0], fg="Black", font=("Times", 18))
        Pdt_svm.grid(row=5, padx=10, column=3, sticky=W)

        acc_svm = Label(root, text=f"{accuracy * 100:.2f}%", fg="Black", font=("Times", 12))
        acc_svm.grid(row=6, padx=10, column=3, sticky=W)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

    def conf_svm():
        # Improved confusion matrix display
        cm = confusion_matrix(Ytest_encoded, predicted_values, labels=list(label_to_num.values()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_to_num.keys()))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix for Linear SVM")
        plt.show()

    def rep_svm():
        report2 = classification_report(Ytest_encoded, predicted_values, target_names=list(label_to_num.keys()))
        messagebox.showinfo("Linear SVM Crop Prediction Report", report2)

    rp_svm = Button(root, text="Report", command=rep_svm, bg="white", fg="Dark red", width=15, font=("Times new roman", 14))
    rp_svm.grid(row=5, column=4, padx=5, pady=10, sticky=W)

    conf_svm_btn = Button(root, text="Confusion Matrix", command=conf_svm, bg="white", fg="Dark red", width=15, font=("Times new roman", 14))
    conf_svm_btn.grid(row=6, column=4, padx=5, pady=10, sticky=W)

# def func_RF():
#     from sklearn.ensemble import RandomForestClassifier
#     RF = RandomForestClassifier(n_estimators=20, random_state=0)
#     RF.fit(Xtrain, Ytrain)

#     predicted_values = RF.predict(Xtest)

#     x = metrics.accuracy_score(Ytest, predicted_values)
#     acc.append(x)
#     model.append('RF')

#     N = float(nty_N.get())
#     P = float(nty_P.get())
#     K = flt(nty_K.get())                 
#     Temperature = float(nty_T.get())
#     Humidity = float(nty_H.get())
#     ph = float(nty_Ph.get())
#     Rainfall = float(nty_R.get())

#     l=[]
#     l.append(N)
#     l.append(P)
#     l.append(K)
#     l.append(Temperature)
#     l.append(Humidity)
#     l.append(ph)
#     l.append(Rainfall)
#     data=[l]

#     #data = np.array([[28,76,82,20.56601874,14.25803981,6.654425315,83.75937135]])
#     prediction = RF.predict(data)

#     Pdt_rf = Label(root, text=prediction,fg="Black" )
#     Pdt_rf.config(font=("Times", 18))
#     Pdt_rf.grid(row=8, padx=10, column=3,sticky=W)

#     acc_rf = Label(root, text= x*100,fg="Black" )
#     acc_rf.config(font=("Times", 12))
#     acc_rf.grid(row=9, padx=10, column=3,sticky=W)

        
#     def conf_rf():
#             ConfusionMatrixDisplay.from_estimator(RF, Xtest, Ytest, cmap = plt.get_cmap('Blues'))  
#             plt.show()

#     def rep_rf():
#         report2 = classification_report(Ytest,predicted_values)
#         messagebox.showinfo("RF Crop Prediction Report", report2)

#     rp_rf = Button(root, text="Report", command=rep_rf,bg="white",fg="Dark red", width=15)
#     rp_rf.config(font=("Times new roman", 14))
#     rp_rf.grid(row=8, column=4,padx=5,pady=10, sticky=W)

#     conf_rf = Button(root, text="Confusion Matrix", command=conf_rf,bg="white",fg="Dark red", width=15)
#     conf_rf.config(font=("Times new roman", 14))
#     conf_rf.grid(row=9, column=4,padx=5,pady=10, sticky=W)

def refresh():
    root.destroy()
    import crop_predictor.py

def back():
    root.destroy()
    import swobaliApp.py

def validate(P):       
    if P.isdigit():
        if int(P) == 0 or int(P) <= 300:
            return True
        else:
            messagebox.showerror("showerror", "Value cannot be greater than 300")
            return False
    else:
        messagebox.showerror("showerror", "It was not a Number. Please enter numeric value.")
        return False

root = Tk()
#root.geometry("400*200")
root.title("SwoBali")
#root.configure(background='white')

img = Image.open("images/swobalilogo.png")
logo = img.resize((80, 100), Image.LANCZOS)
logo = ImageTk.PhotoImage(logo)
label = Label(root, image = logo)
label.image = logo
label.grid(row=1, column=0, columnspan=12,pady=20, padx=100)

head1 = Label(root, justify=LEFT, text="Krishi", fg="Dark green" )
head1.config(font=("Elephant", 32,))
head1.grid(row=1, column=0, columnspan=12, padx=100)
  
head2 = Label(root, justify=LEFT, text="Crop Predictor", fg="black" )
head2.config(font=("Aharoni", 22))
head2.grid(row=2, column=0, columnspan=12, padx=100)

lbl_N = Label(root, text="Nitrogen", fg="Black")
lbl_N.config(font=("Times", 18, "bold"))
lbl_N.grid(row=4, column=1, pady=10, padx=10, sticky=W)

lbl_P = Label(root, text="Phosphorous", fg="Black")
lbl_P.config(font=("Times", 18, "bold"))
lbl_P.grid(row=5, column=1, pady=10, padx=10, sticky=W)

lbl_K = Label(root, text="Potassium", fg="Black")
lbl_K.config(font=("Times", 18, "bold"))
lbl_K.grid(row=6, column=1, pady=10, padx=10, sticky=W)

lbl_T = Label(root, text="Temperature", fg="Black")
lbl_T.config(font=("Times", 18, "bold"))
lbl_T.grid(row=7, column=1, pady=10, padx=10, sticky=W)

lbl_H = Label(root, text="Humidity", fg="Black")
lbl_H.config(font=("Times", 18, "bold"))
lbl_H.grid(row=8, column=1, pady=10, padx=10, sticky=W)

lbl_Ph = Label(root, text="Ph", fg="Black")
lbl_Ph.config(font=("Times", 18, "bold"))
lbl_Ph.grid(row=9, column=1, pady=10, padx=10, sticky=W)

lbl_R = Label(root, text="Rainfall", fg="Black")
lbl_R.config(font=("Times", 18, "bold"))
lbl_R.grid(row=10, column=1, pady=10, padx=10, sticky=W)

nty_N = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_N.grid(row=4, column=2, padx=10, sticky=W)

nty_P = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_P.grid(row=5, column=2, padx=10, sticky=W)

nty_K = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_K.grid(row=6, column=2, padx=10, sticky=W)

nty_T = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_T.grid(row=7, column=2, padx=10, sticky=W)

nty_H = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_H.grid(row=8, column=2, padx=10, sticky=W)

nty_Ph = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_Ph.grid(row=9, column=2, padx=10, sticky=W)

nty_R = Entry(root, validate="key", validatecommand=(root.register(validate), '%P'))
nty_R.grid(row=10, column=2, padx=10, sticky=W)

ls = Button(root, text="Predict using Linear SVM", command=func_LinearSVM, bg="White", fg="Dark red", width=15, padx=80)
ls.config(font=("Times new roman", 16))
ls.grid(row=4, column=3, columnspan=6, padx=10, pady=10, sticky=W)

# lr = Button(root, text="Predict using RF", command=func_RF, bg="white", fg="Dark red", width=15, padx=80)
# lr.config(font=("Times new roman", 16))
# lr.grid(row=7, column=3, columnspan=6, padx=10, pady=10, sticky=W)

ref = Button(root, text="Refresh", command=refresh, bg="grey", fg="Dark green", width=15)
ref.config(font=("Times new roman", 16))
ref.grid(row=10, column=4, padx=10, pady=10, sticky=W)

bck = Button(root, text="Go back", command=back, bg="grey", fg="Dark green", width=15)
bck.config(font=("Times new roman", 16))
bck.grid(row=10, column=3, padx=10, pady=10, sticky=W)

root.mainloop()
