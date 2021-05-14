#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf 
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train.shape


# In[ ]:


import matplotlib.pyplot as  plt 
get_ipython().run_line_magic('matplotlib', 'inline')
fig, axs=plt.subplots(4,4,figsize=(10,15))
plt.gray()

for i,ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.plot()
    ax.set_title('the number is {}'.format(y_train[i]),loc='center')
    


# In[ ]:


x_train= x_train.reshape(x_train.shape[0],28,28,1)
x_test= x_test.reshape(x_test.shape[0],28,28,1)
input_shape=(28,28,1)


# In[ ]:


x_train= x_train.astype('float32')
x_test=x_test.astype('float32')
x_train /= 225
x_test /=225
print('the {}'.format(x_train.shape))
print('the {}'.format(x_train.shape[0]))
print('the {}'.format(x_test.shape[0]))


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(32,3,3,input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation=(tf.nn.relu)))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
          


# In[ ]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy'
             )
model.fit(x=x_train,y=y_train, epochs=1)


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:




