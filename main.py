from numpy import exp, array, random, dot, asarray


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((100, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(asarray(inputs, self.synaptic_weights)))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([
        [63,1,1,145,233,1,2,150,0,2.3,3,0,6],
        [67,1,4,160,286,0,2,108,1,1.5,2,3,3],
        [67,1,4,120,229,0,2,129,1,2.6,2,2,7],
        [37,1,3,130,250,0,0,187,0,3.5,3,0,3],
        [41,0,2,130,204,0,2,172,0,1.4,1,0,3],
        [56,1,2,120,236,0,0,178,0,0.8,1,0,3],
        [62,0,4,140,268,0,2,160,0,3.6,3,2,3],
        [57,0,4,120,354,0,0,163,1,0.6,1,0,3],
        [63,1,4,130,254,0,2,147,0,1.4,2,1,7],
        [53,1,4,140,203,1,2,155,1,3.1,3,0,7],
        [57,1,4,140,192,0,0,148,0,0.4,2,0,6],
        [56,0,2,140,294,0,2,153,0,1.3,2,0,3],
        [56,1,3,130,256,1,2,142,1,0.6,2,1,6],
        [44,1,2,120,263,0,0,173,0,0,1,0,7],
        [52,1,3,172,199,1,0,162,0,0.5,1,0,7],
        [57,1,3,150,168,0,0,174,0,1.6,1,0,3],
        [48,1,2,110,229,0,0,168,0,1,3,0,7],
        [54,1,4,140,239,0,0,160,0,1.2,1,0,3],
        [48,0,3,130,275,0,0,139,0,0.2,1,0,3],
        [49,1,2,130,266,0,0,171,0,0.6,1,0,3],
        [64,1,1,110,211,0,2,144,1,1.8,2,0,3],
        [58,0,1,150,283,1,2,162,0,1,1,0,3],
        [58,1,2,120,284,0,2,160,0,1.8,2,0,3],
        [58,1,3,132,224,0,2,173,0,3.2,1,2,7],
        [60,1,4,130,206,0,2,132,1,2.4,2,2,7],
        [50,0,3,120,219,0,0,158,0,1.6,2,0,3],
        [58,0,3,120,340,0,0,172,0,0,1,0,3],
        [66,0,1,150,226,0,0,114,0,2.6,3,0,3],
        [43,1,4,150,247,0,0,171,0,1.5,1,0,3],
        [40,1,4,110,167,0,2,114,1,2,2,0,7],
        [69,0,1,140,239,0,0,151,0,1.8,1,2,3],
        [60,1,4,117,230,1,0,160,1,1.4,1,2,7],
        [64,1,3,140,335,0,0,158,0,0,1,0,3],
        [59,1,4,135,234,0,0,161,0,0.5,2,0,7],
        [44,1,3,130,233,0,0,179,1,0.4,1,0,3],
        [42,1,4,140,226,0,0,178,0,0,1,0,3],
        [43,1,4,120,177,0,2,120,1,2.5,2,0,7],
        [57,1,4,150,276,0,2,112,1,0.6,2,1,6],
        [55,1,4,132,353,0,0,132,1,1.2,2,1,7],
        [61,1,3,150,243,1,0,137,1,1,2,0,3],
        [65,0,4,150,225,0,2,114,0,1,2,3,7],
        [40,1,1,140,199,0,0,178,1,1.4,1,0,7],
        [71,0,2,160,302,0,0,162,0,0.4,1,2,3],
        [59,1,3,150,212,1,0,157,0,1.6,1,0,3],
        [61,0,4,130,330,0,2,169,0,0,1,0,3],
        [58,1,3,112,230,0,2,165,0,2.5,2,1,7],
        [51,1,3,110,175,0,0,123,0,0.6,1,0,3],
        [50,1,4,150,243,0,2,128,0,2.6,2,0,7],
        [65,0,3,140,417,1,2,157,0,0.8,1,1,3],
        [53,1,3,130,197,1,2,152,0,1.2,3,0,3],
        [41,0,2,105,198,0,0,168,0,0,1,1,3],
        [65,1,4,120,177,0,0,140,0,0.4,1,0,7],
        [44,1,4,112,290,0,2,153,0,0,1,1,3],
        [44,1,2,130,219,0,2,188,0,0,1,0,3],
        [60,1,4,130,253,0,0,144,1,1.4,1,1,7],
        [54,1,4,124,266,0,2,109,1,2.2,2,1,7],
        [50,1,3,140,233,0,0,163,0,0.6,2,1,7],
        [41,1,4,110,172,0,2,158,0,0,1,0,7],
        [54,1,3,125,273,0,2,152,0,0.5,3,1,3],
        [51,1,1,125,213,0,2,125,1,1.4,1,1,3],
        [51,0,4,130,305,0,0,142,1,1.2,2,0,7],
        [46,0,3,142,177,0,2,160,1,1.4,3,0,3],
        [58,1,4,128,216,0,2,131,1,2.2,2,3,7],
        [54,0,3,135,304,1,0,170,0,0,1,0,3],
        [54,1,4,120,188,0,0,113,0,1.4,2,1,7],
        [60,1,4,145,282,0,2,142,1,2.8,2,2,7],
        [60,1,3,140,185,0,2,155,0,3,2,0,3],
        [54,1,3,150,232,0,2,165,0,1.6,1,0,7],
        [59,1,4,170,326,0,2,140,1,3.4,3,0,7],
        [46,1,3,150,231,0,0,147,0,3.6,2,0,3],
        [65,0,3,155,269,0,0,148,0,0.8,1,0,3],
        [67,1,4,125,254,1,0,163,0,0.2,2,2,7]
    ])
    training_set_outputs = array([[0,1,1,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,1,1,1,0,1,1,0,1]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering new situation [65,1,4,110,248,0,2,158,0,0.6,1,2,6] -> ?: ")
    print(neural_network.think(array([65,1,4,110,248,0,2,158,0,0.6,1,2,6])))
