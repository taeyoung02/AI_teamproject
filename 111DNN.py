from keras.models import Sequential
from keras.layers import Dense

weight = {0:0.1, 1:99}

def DNN1(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=29, activation='relu',input_dim=29))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad',weighted_metrics=weight)
    #배치 사이즈는 가중치 갱신을 몇번에 한번 하는지.
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN2(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=29, activation='relu',input_dim=29))
    NN.add(Dense(units=29, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad',weighted_metrics=weight)
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN3(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=29, activation='relu',input_dim=29))
    NN.add(Dense(units=29, activation='relu'))
    NN.add(Dense(units=29, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad',weighted_metrics=weight)
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN4(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=15, activation='relu',input_dim=29))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    #배치 사이즈는 가중치 갱신을 몇번에 한번 하는지.
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN5(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=15, activation='relu',input_dim=29))
    NN.add(Dense(units=15, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN6(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=15, activation='relu',input_dim=29))
    NN.add(Dense(units=15, activation='relu'))
    NN.add(Dense(units=15, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN7(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=45, activation='relu',input_dim=29))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    #배치 사이즈는 가중치 갱신을 몇번에 한번 하는지.
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN8(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=45, activation='relu',input_dim=29))
    NN.add(Dense(units=45, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN9(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=45, activation='relu',input_dim=29))
    NN.add(Dense(units=45, activation='relu'))
    NN.add(Dense(units=45, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN10(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=29, activation='relu',input_dim=29))
    NN.add(Dense(units=29, activation='relu'))
    NN.add(Dense(units=29, activation='relu'))
    NN.add(Dense(units=29, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))

def DNN11(x_train,y_train,x_test,y_test):
    NN = Sequential()

    NN.add(Dense(units=45, activation='relu',input_dim=29))
    NN.add(Dense(units=45, activation='relu'))
    NN.add(Dense(units=45, activation='relu'))
    NN.add(Dense(units=45, activation='relu'))
    NN.add(Dense(units=2, activation='softmax'))

    NN.compile(loss='mse',optimizer='adagrad')
    NN.fit(x_train,y_train,epochs=1,batch_size=100)
    print(NN.evaluate(x_test,y_test))
