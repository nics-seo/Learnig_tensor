
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[3]:


X = [1, 2, 3]
Y = [1, 2, 3]
W = tf.placeholder(tf.float32)
hypothesis = W * X


# In[4]:


cost = tf.reduce_mean(tf.square(hypothesis - Y))


# In[5]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[6]:


W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W : feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)


# In[7]:


plt.plot(W_val, cost_val)
plt.show()

