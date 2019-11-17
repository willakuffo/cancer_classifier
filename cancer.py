
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense
from keras.models import Sequential
import seaborn as sns
import matplotlib.pyplot as plt
import time

df = pd.read_csv('cancer.csv',delim_whitespace = True,header = None)
sns.set(color_codes = True)
sns.pairplot(df)
#plt.show()
test_set = df[512:]

test_labels = test_set[2]

test_set = pd.concat([test_set.iloc[:,3:-1],test_set[1]],axis = 1)
labels = df[2]
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.fit_transform(labels)
train_labels = np_utils.to_categorical(labels)

encoder.fit(test_labels)
test_labels = encoder.fit_transform(test_labels)
test_labels = np_utils.to_categorical(test_labels)

train_samples = pd.concat([df.iloc[:,3:-1],df[1]],axis = 1)
sc = MinMaxScaler()
train_samples = sc.fit_transform(train_samples)
test_set = sc.fit_transform(test_set)


print(train_samples.shape,train_labels.shape)

initial_time = time.time()
model = Sequential()
model.add(Dense(units = train_samples.shape[1],input_dim =  train_samples.shape[1],activation = 'relu'))
model.add(Dense(units = 64,activation = 'relu'))
model.add(Dense(units = train_labels.shape[1],activation = 'softmax'))
 
model.summary()

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
fit_time = time.time()
model.fit(x = train_samples,y = train_labels,validation_split= 0.1,epochs = 150)
final_time = time.time()
predictions = model.predict_classes(train_samples,batch_size=10)
preds_probs = model.predict_proba(train_samples,batch_size= 10)

run_time = final_time-initial_time
print('model time:',run_time,'fit_train time:',final_time-fit_time)
print(preds_probs)

from sklearn.metrics import confusion_matrix
import itertools
print(type(test_labels))

def model_eval(class_predictions = None ,marking_scheme = None,are_probabs = False ):
    scheme = list(marking_scheme)
    preds = list(class_predictions)
    print(type(preds),type(scheme))
    if are_probabs == False:
        if preds!=None and scheme!=None:
            corrects = 0
            
            for i in range(len(preds)):
                print(preds[i],scheme[i])
                if preds[i] == scheme[i][1]:
                    corrects = corrects+1
            score = corrects/len(preds)
    else:
        if preds!=None and scheme!=None:
            corrects = 0
            for i in range(len(preds)):
                
                if abs(preds[i][1] - scheme[i][1])<0.5:
                    corrects = corrects+1
                else:
                    print(preds[i],scheme[i])
            score = corrects/len(preds)


            print(score)


#model_eval(class_predictions = predictions,marking_scheme= train_labels)


model_eval(class_predictions = preds_probs,marking_scheme= train_labels,are_probabs= True)
#cm = confusion_matrix(test_labels,predictions)
#print(cm)

def con_mat(cm,classes,normalize = False,title = 'confusion_matrix',cmap = plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks =np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation = 45)
    plt.yticks(tick_marks,classes)

    if normalize:
        cm = cm.astype('float')/cm.sum(axis =1)[:,np.newaxis]
    print(cm)
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment = 'center',color = 'white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('predicted')
    plt.show()

#cmlabels = ['malign','benign']
#con_mat(cm,cmlabels,title = 'con mat')