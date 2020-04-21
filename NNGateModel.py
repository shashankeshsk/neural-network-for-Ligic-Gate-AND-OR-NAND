
import numpy as np
import matplotlib.pyplot as plt


train_set_x = np.array([[0,0,1,1],[0,1,0,1]]) 
train_set_yAND = np.array([[0,0,0,1]])#AND
train_set_yOR = np.array([[0,1,1,1]])#OR
train_set_yNAND = np.array([[1,1,1,0]])#NAND
#test_set_x = np.array([[1],[1]])
#test_set_y = np.array([[1]])
#train_set_y.shape


def hardlim(z):
    
    s = (z>=0).astype(float)
    
    return s



def initialize_with_zeros(dim):
    
    w = np.zeros((dim,1))
    b = 0.0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b



def propagate(w, b, X, Y):
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = hardlim(np.dot(w.T,X)+b)                                    # compute activation
    cost = np.sum((A-Y)**2)/2                         # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X,(A-Y).T)
    db = np.sum((A-Y))

    cost = np.squeeze(cost)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 1 == 0:
            costs.append(cost)
        
        # Print the cost every 1 training iterations
        if print_cost and i % 1 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs



def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = hardlim(np.dot(w.T,X)+b)
    
    Y_prediction = A
    
	#assert(Y_prediction.shape == (1, m))
    
    return Y_prediction



def model(X_train, Y_train, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict train set examples
    Y_prediction_train = predict(w, b, X_train)

    # Print train Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('w: {}'.format(w))
    print('b: {}'.format(b))
    print('Y_prediction_train: {}'.format(Y_prediction_train))
    print('Y_train: {}'.format(Y_train))
    d = {"costs": costs,
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


print("---------AND Neural Network Model----------")
dAND = model(train_set_x, train_set_yAND, num_iterations = 10, learning_rate = 0.01, print_cost = True)
# Plot learning curve (with costs)
costs = np.squeeze(dAND['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(dAND["learning_rate"]))
plt.show()
print("----------OR Neural Network Model-----------")
dOR = model(train_set_x, train_set_yOR, num_iterations = 10, learning_rate = 0.01, print_cost = True)
# Plot learning curve (with costs)
costs = np.squeeze(dOR['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations ')
plt.title("Learning rate =" + str(dOR["learning_rate"]))
plt.show()
print("-----------NAND Neural Network Model---------")
dNAND = model(train_set_x, train_set_yNAND, num_iterations = 10, learning_rate = 0.01, print_cost = True)
# Plot learning curve (with costs)
costs = np.squeeze(dNAND['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations ')
plt.title("Learning rate =" + str(dNAND["learning_rate"]))
plt.show()