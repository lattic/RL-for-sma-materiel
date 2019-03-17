#数据预处理
import xlrd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
try:
    volt_N=np.load('volt_N.npy')
    defor_N=np.load('defor_N.npy')
except:
    volt=np.zeros((5,25))
    defor=np.zeros((5,25))
    da=xlrd.open_workbook("data.xlsx")
    table=da.sheets()[0]
    for i in range(10):
        if(i%2==0):
            volt[int(i/2),:]=table.col_values(i)[4:]
        else:
            defor[i//2,:]=-np.array(table.col_values(i)[4:])+table.col_values(i)[1]
    #normalization
    volt_N=(volt-np.min(volt))/(np.max(volt)-np.min(volt))
    defor_N=(defor-np.min(defor))/(np.max(defor)-np.min(defor))
    np.save("volt_N.npy",volt_N)
    np.save("defor_N.npy",defor_N)
class SMA_model(object):
    def __init__(self,n_steps,cell_size,batch_size,n_input=1,n_output=1):
        self.n_steps = n_steps
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.n_input=n_input
        self.n_output=n_output
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps], name='xs')#(batch_size,n_steps*n_inputs)
            self.ys = tf.placeholder(tf.float32, [None, n_steps], name='ys')#(batch_size,n_steps*n_outputs)

        self.weights = {
            'in': tf.Variable(tf.random_normal([self.n_input, self.cell_size])),  # (1, cell_size)
            'out': tf.Variable(tf.random_normal([self.cell_size, self.n_output]))  # (cell_size, 1)
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[cell_size])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
        }
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(0.01, self.global_step, 100, 0.96, staircase=False)
            self.pred = self.lstm()
            self.label=tf.reshape(self.ys,[-1,self.n_output])
            self.mse_cost = tf.reduce_mean(tf.square(self.pred-self.label))#+tf.contrib.layers.l2_regularizer(0.005)(self.weights)
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss=self.mse_cost,
                                                                                           global_step=self.global_step)

    def lstm(self):
        X = tf.reshape(self.xs, [-1, self.n_input])   #(batch_size*n_step,n_input)
        X_in = tf.matmul(X, self.weights['in']) + self.biases['in']     #(batch_size*n_step, cell_size)
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.cell_size]) #(batch_size, n_step, cell_size)
        # 定义LSTMcell
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
        self.init_state = self.lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(self.lstm_cell, X_in, initial_state=self.init_state, time_major=False)
        outputs_=tf.reshape(outputs,[-1,self.cell_size])
        results = tf.sigmoid(tf.matmul(outputs_, self.weights['out']) + self.biases['out'])
        return results #（batch_size*n_steps,n_output)
if __name__!='main':
    model=SMA_model(n_steps=25,cell_size=100,batch_size=5,n_input=1,n_output=1)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    """Training"""
    yy=np.zeros((5,25))
    N=10000
    for i in range(N):
        feed_dict={
            model.xs:  volt_N,
            model.ys:  defor_N
           # model.init_state: tf.contrib.rnn.BasicLSTMCell.zero_state(model.batch_size, dtype=tf.float32)
        }
       # print("xxx")
        _,cost,pred=sess.run([model.train_op,model.mse_cost,model.pred],feed_dict=feed_dict)
        if i%10==0:
            print('cost',cost)
        if i==N-1:
            yy=pred.reshape((5,25))
    x=np.array(range(25)).reshape(25)
    plt.figure(1)
    plt.plot(x,defor_N[0,:],'r',label='ground truth')
    print(defor_N[0,:])
    plt.plot(x,yy[0,:],'y',label='predict by lstm')
    plt.legend()

    plt.figure(2)
    plt.plot(x, defor_N[1, :], 'r', label='ground truth')
    print(defor_N[0, :])
    plt.plot(x, yy[1,:], 'y', label='predict by lstm')
    plt.legend()
    plt.show()

