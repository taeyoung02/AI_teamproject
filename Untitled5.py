#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
data = pd.read_csv('C:\\Users\\owner\\Downloads\\data\\X_train.csv')


# In[94]:


y_train = pd.read_csv('C:\\Users\\owner\\Downloads\\data\\y_train.csv')


# In[96]:


y_train


# In[97]:


y_train=y_train.iloc[:,1:]


# In[98]:


y_train


# In[37]:


data


# In[38]:


data = data.iloc[:, 2:]


# In[39]:


data


# In[40]:


data.shape


# In[41]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np


# In[151]:


pca = PCA(random_state=777)
X_p = pca.fit_transform(data)

import pandas as pd
v_ratio = pd.Series(np.cumsum(pca.explained_variance_ratio_)) # 누적 분산을 나타냄


# In[157]:


v_ratio


# In[156]:


# StandardScaler	기본 스케일. 평균과 표준편차 사용   
# MinMaxScaler	최대/최소값이 각각 1, 0이 되도록 스케일링   
# MaxAbsScaler	최대절대값과 0이 각각 1, 0이 되도록 스케일링   
# RobustScaler	중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
#  

# In[42]:


# 후에 아웃라이어 확인
from sklearn import preprocessing
x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_train = pd.DataFrame(x_scaled)


# In[43]:


x_train[:10]


# In[44]:


x_train.shape


# In[28]:


val = pd.read_csv('C:\\Users\\owner\\Downloads\\data\\validation.csv')


# In[29]:


val


# In[45]:


x_val = val.iloc[:, 2:-1]


# In[46]:


y_val = val.iloc[:, -1:]


# In[47]:


x_val


# In[49]:


y_val


# In[103]:


from keras import backend as K
def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


# In[17]:


from keras import models, layers
from keras import optimizers


# In[77]:


model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (29,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[106]:


model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy', recall])


# In[100]:


X_train = np.asarray(x_train).astype('float32').reshape((-1,29))
X_val = np.asarray(x_val).astype('float32').reshape((-1,29))
Y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
Y_val = np.asarray(y_val).astype('float32').reshape((-1,1))


# In[101]:


Y_train.shape


# In[107]:


history = model.fit(X_train,
                   Y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data = (X_val, Y_val))


# In[110]:


import matplotlib.pyplot as plt


history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()



plt.show()


# In[111]:


from keras import regularizers


# In[129]:


model2 = models.Sequential()
model2.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation = 'relu', input_shape = (29,)))
model2.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))
model2.add(layers.Dense(1, activation='sigmoid'))


# In[130]:


model2.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy', recall])


# In[131]:


history2 = model2.fit(X_train,
                   Y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data = (X_val, Y_val))


# In[132]:


import matplotlib.pyplot as plt


history_dict = history2.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()



plt.show()


# In[126]:


model2.compile(optimizer='Adam',
             loss='binary_crossentropy',
             metrics=['accuracy', recall])


# In[127]:


history3 = model2.fit(X_train,
                   Y_train,
                   epochs=20,
                   batch_size=512,
                   validation_data = (X_val, Y_val))


# In[128]:


import matplotlib.pyplot as plt


history_dict = history3.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()



plt.show()


# In[133]:


import matplotlib.pyplot as plt


history_dict = history3.history
loss = history_dict['recall']
val_loss = history_dict['val_recall']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training recall')
plt.plot(epochs, val_loss, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.xlabel('Epochs')
plt.ylabel('recall')
plt.legend()



plt.show()


# In[134]:


import matplotlib.pyplot as plt


history_dict = history2.history
loss = history_dict['recall']
val_loss = history_dict['val_recall']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training recall')
plt.plot(epochs, val_loss, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.xlabel('Epochs')
plt.ylabel('recall')
plt.legend()



plt.show()


# In[135]:


import matplotlib.pyplot as plt


history_dict = history.history
loss = history_dict['recall']
val_loss = history_dict['val_recall']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training recall')
plt.plot(epochs, val_loss, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.xlabel('Epochs')
plt.ylabel('recall')
plt.legend()



plt.show()


# In[136]:


X_test = pd.read_csv('C:\\Users\\owner\\Downloads\\data\\X_test.csv')
y_test = pd.read_csv('C:\\Users\\owner\\Downloads\\data\\y_test.csv')


# In[137]:


X_test=X_test.iloc[:,2:]


# In[140]:


X_test


# In[138]:


predict1=model2.predict(X_test)


# In[139]:


predict1


# In[ ]:




