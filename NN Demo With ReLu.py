from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        #I'm going to make a seed, for a random number generator.
        #Basically, this generates a random number, but will generate the same
        #random number every time.
        random.seed(1)

        #We're going to make one neuron with 3 input connections and 1 output.
        #SO that means we need to assign random weights to a 3x1 matrix
        #All the numbers are between -1 and 1, with a mean of 0.
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    #We'll be using a sigmoid function here, which forms an S shaped curve
    #We pass the weighted sum of the inputs through this function
    #to normalise them between 0 and 1
    def __sigmoid(self, x):
        return 1 /(1 + exp(-x))

    #Gradient of a sigmoid curve, tells us how confident we are for the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, train_inputs, train_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #pass the training set through our neural net
            output = self.predict(train_inputs)

            #We'll next calculate the error
            error = train_outputs - output

            #multiply the error by input and again by the gradient of the
            #sigmoid curve
            adjustment = dot(train_inputs.T, error * self.__sigmoid_derivative(output))

            #And then we adjust the weights
            self.synaptic_weights += adjustment
    
    def predict(self, inputs):
        #pass inputs through our neural network (our single neuron)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':

    #Initialising a perceptron
    nn = NeuralNetwork()

    print ("Here will be some random weights")
    print (nn.synaptic_weights)

    #This is going to be the training set
    #There are 4 examples, each with 3 inputs and 1 output.
    train_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    train_outputs = array ([[0,1,1,0]]).T

    #Using a neural network to train the traning set.
    #Training it 10,000 times while making adjustments per test
    nn.train(train_inputs, train_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (nn.synaptic_weights)

    #Testing the neural network
    print ("Prediction: ")
    print (nn.predict(array([[1,0,0]])))
