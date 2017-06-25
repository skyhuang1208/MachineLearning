import numpy as np
from scipy.special import expit

class classification():
    def __init__(self, nf, hidden_sl, lrate=  60.0, penalty= 0.0001, maxiter=50000, tol= 1e-6):
        # initialize variables
        self.isfitted= False # fit func not called
        self.nlayers= len(hidden_sl)+2
        self.nhidden= len(hidden_sl)
        self.sl=  np.copy(hidden_sl)
        self.nfeature= nf
        self.penalty= penalty
        self.learning_rate= lrate
        self.maxiter= maxiter
        self.tol= tol
        self.cost= []

        # initialize neurons
        self.neurons= [ [0 for i in range(sl)] for sl in hidden_sl]
        self.neurons.insert(0, []) # keep same layer index

        # initialize weight arrays
        # given x normalized to (0, 1), weights is set [-1/sqrt(nf), 1/sqrf(nf)]
        self.coeffs= []
        rr= 4/np.sqrt(nf) # random range
        self.coeffs.append( 2*rr*np.random.random( (hidden_sl[0],nf) )-rr )
        for l in range(0, self.nhidden-1): # hidden layers
            rr= 4/np.sqrt(hidden_sl[l]) # random range
            self.coeffs.append( 2*rr*np.random.random( (hidden_sl[l+1], hidden_sl[l]) )-rr )
        rr= 4/np.sqrt(hidden_sl[-1]) # random range
        self.coeffs.append( 2*rr*np.random.random( (1,hidden_sl[-1]) )-rr )
        # initialize bias vectors
        self.bias= [ 2*rr*np.random.random(sl)-rr for sl in hidden_sl]
        self.bias.append( 2*rr*np.random.random()-rr )

        # initialize delta arrays (derivative of cost)
        self.delta= []
        self.delta.append( np.zeros((hidden_sl[0],nf)) )
        for l in range(0, self.nhidden-1): # hidden layers
            self.delta.append( np.zeros((hidden_sl[l+1], hidden_sl[l])) )
        self.delta.append( np.zeros((1,hidden_sl[-1])) )
        # initialize bias delta vectors
        self.dbias= [np.zeros(sl) for sl in hidden_sl]
        self.dbias.append(0)

    def fit(self, x, y):
        if len(x) != len(y): exit("Error:(fit) len of x != y: %d, %d" % (len(x), len(y))) # check

        # initializing
        self.isfitted= True
        x= np.array(x)
        y= np.array(y)
        self.xmean= np.mean(x, axis=0)
        self.xstd= np.std(x, axis=0)
        x= np.array([(xi-self.xmean)/self.xstd for xi in x]) # normalize

        (n,p)= x.shape # N of samples, features
        if self.nfeature != p: exit("Error:(fit) N of features != size of x: %d, %d" % (self.nfeature, p))

        # start fitting
        for t in range(self.maxiter):
            for i in range(len(self.delta)):self.delta[i][:]= 0. # initializing
            for i in range(len(self.dbias)-1):self.dbias[i][:]= 0.; self.dbias[-1]= 0
            cost= 0

            for i in range(n): # loop tho samples
                output= self.__forward(x[i,:])
                errorOUT= output - y[i] # output layer error
                self.__backward(errorOUT, x[i,:])
                cost -= self.__calcost(y[i], output)/n
            cost += self.penalty/2/n*sum([np.sum(np.square(c)) for c in self.coeffs])

            if(t>10 and abs(cost-self.cost[-1])<self.tol): break
            self.cost.append(cost)
            self.__graddesc(n)

            print(cost[0], t)

    __calcost= lambda self, y, y_hat: y*np.log(y_hat)+(1-y)*np.log(1-y_hat)

    def predict(self, x):
        if not self.isfitted: exit("Error(predict) train the model using fit first")
        if len(x[0]) != self.nfeature: exit("Error:(predict) x size incorrect: %d %d", len(x), self.nfeature)

        x= np.array([(xi-self.xmean)/self.xstd for xi in x]) # normalize
        y_hat= [self.__forward(xi) for xi in x]
        return np.array(y_hat)

    # private
    def __forward(self, x_):
        # neuron[l+1]= logistic( weight[l] * neuron[l] )
        self.neurons[1]= expit( np.dot(self.coeffs[0],x_)+self.bias[0] )
        for l in range(1,self.nlayers-2):
            self.neurons[l+1]= expit( np.dot(self.coeffs[l],self.neurons[l])+self.bias[l] )
        return expit( np.dot(self.coeffs[-1],self.neurons[-1])+self.bias[-1] )

    def __backward(self, errorOUT, x_):
        # error[l] = ( weight[l] * error[l+1] ) .* ( neuron[l] .* (1-neuron[l]) )
        # delta[l] += error[l+1] (multiply) neuron[l]
        error= [ np.zeros(sl) for sl in self.sl ]; error.insert(0, []) #initialize (same layer index)

        self.delta[-1] += np.outer(errorOUT, self.neurons[-1])
        self.dbias[-1] += errorOUT
        term1= (self.coeffs[-1].T)*errorOUT
        term2= np.multiply(self.neurons[-1],(1-self.neurons[-1]))
        error[-1]= np.multiply( term1.flatten(), term2.flatten() )
        for l in range(self.nlayers-3, 0, -1): # ignore l=0 since no error for input layer
            term1= np.dot( (self.coeffs[l].T), error[l+1] )
            term2= np.multiply( self.neurons[l], (1-self.neurons[l]) )
            error[l]= np.multiply( term1.flatten(), term2.flatten() )
            self.delta[l] += np.outer(error[l+1],self.neurons[l])
            self.dbias[l] += error[l+1]
        self.delta[0] += np.outer(error[1],x_)
        self.dbias[0] += error[1]

    def __graddesc(self, n): # Gradient descent
        for i, _ in enumerate(self.coeffs):
            self.coeffs[i] -= self.learning_rate * (self.delta[i]/n + self.penalty*self.coeffs[i])
#        for i in range(len(self.delta)): self.delta[i] /= nsamples
#        for i in range(len(self.delta)): self.delta[i] += self.penalty*self.coeffs[i]
#        for i in range(len(self.dbias)): self.dbias[i] /= nsamples
