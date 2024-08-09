#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Helper function to generate one-hot encoded vector
std::vector<float> oneHotEncode(int value, int maxValue) {
    std::vector<float> vec(maxValue, 0.0);
    if (value >= 0 && value < maxValue) {
        vec[value] = 1.0;
    }
    return vec;
}

// Simple neural network class
class SimpleNeuralNetwork {
public:
    SimpleNeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        // Initialize weights and biases
        weights1.resize(inputSize, std::vector<float>(hiddenSize, 0.0));
        biases1.resize(hiddenSize, 0.0);
        weights2.resize(hiddenSize, std::vector<float>(outputSize, 0.0));
        biases2.resize(outputSize, 0.0);
        initializeWeights();
    }

    // Forward propagation
    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> hidden = activate(linearTransform(input, weights1, biases1));
        return linearTransform(hidden, weights2, biases2); // No activation on the final layer
    }

    // Training function
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalError = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Forward propagation
                std::vector<float> hidden = activate(linearTransform(inputs[i], weights1, biases1));
                std::vector<float> outputs = linearTransform(hidden, weights2, biases2);

                // Compute errors
                std::vector<float> outputErrors(targets[i].size(), 0.0);
                for (size_t j = 0; j < outputErrors.size(); ++j) {
                    outputErrors[j] = targets[i][j] - outputs[j];
                    totalError += outputErrors[j] * outputErrors[j];
                }

                // Backward propagation (simplified)
                updateWeights(inputs[i], hidden, outputs, outputErrors, learningRate);
            }

            if (epoch % 100 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << epoch + 1 << ", Total Error: " << totalError << std::endl;
            }
        }
    }

private:
    std::vector<std::vector<float>> weights1, weights2;
    std::vector<float> biases1, biases2;

    // Initialize weights with random values
    void initializeWeights() {
        std::srand(std::time(nullptr));
        for (auto& row : weights1) {
            std::generate(row.begin(), row.end(), []() { return static_cast<float>(std::rand()) / RAND_MAX; });
        }
        for (auto& row : weights2) {
            std::generate(row.begin(), row.end(), []() { return static_cast<float>(std::rand()) / RAND_MAX; });
        }
    }

    // Linear transformation
    std::vector<float> linearTransform(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
        std::vector<float> output(biases.size(), 0.0);
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                output[j] += input[i] * weights[i][j];
            }
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            output[i] += biases[i];
        }
        return output;
    }

    // Activation function (ReLU)
    std::vector<float> activate(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        std::transform(x.begin(), x.end(), result.begin(), [](float val) { return std::max(0.0f, val); });
        return result;
    }

    // Update weights using simple backpropagation
    void updateWeights(const std::vector<float>& input, const std::vector<float>& hidden, const std::vector<float>& outputs, const std::vector<float>& outputErrors, float learningRate) {
        // Update weights2 and biases2
        for (size_t i = 0; i < weights2.size(); ++i) {
            for (size_t j = 0; j < weights2[i].size(); ++j) {
                weights2[i][j] += learningRate * outputErrors[j] * hidden[i];
            }
        }
        for (size_t i = 0; i < biases2.size(); ++i) {
            biases2[i] += learningRate * outputErrors[i];
        }

        // Compute hidden errors
        std::vector<float> hiddenErrors(hidden.size(), 0.0);
        for (size_t i = 0; i < hiddenErrors.size(); ++i) {
            for (size_t j = 0; j < outputErrors.size(); ++j) {
                hiddenErrors[i] += outputErrors[j] * weights2[i][j];
            }
        }

        // Update weights1 and biases1
        for (size_t i = 0; i < weights1.size(); ++i) {
            for (size_t j = 0; j < weights1[i].size(); ++j) {
                weights1[i][j] += learningRate * hiddenErrors[j] * input[i];
            }
        }
        for (size_t i = 0; i < biases1.size(); ++i) {
            biases1[i] += learningRate * hiddenErrors[i];
        }
    }
};

int main() {
    int maxValue = 31;  // Maximum value in input for one-hot encoding
    SimpleNeuralNetwork nn(maxValue, 10, 1); // 1 output for the count

    // Example input data
    std::vector<int> input = { 10, 20, 15, 30, 40, 10, 30 };
    std::unordered_map<int, int> countMap;

    // Create training data
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
    std::unordered_map<int, int> targetMap;
    for (int num : input) {
        inputs.push_back(oneHotEncode(num, maxValue));
        targetMap[num]++;
    }

    // Create targets from the targetMap
    for (int num : input) {
        std::vector<float> target = { static_cast<float>(targetMap[num]) };
        targets.push_back(target);
    }

    // Train the network
    nn.train(inputs, targets, 2000, 0.001);

    // Predict counts using the trained network
    std::vector<int> test = { 10, 20, 15, 30, 40, 10, 30 };
    for (int num : test) {
        std::vector<float> inputVec = oneHotEncode(num, maxValue);
        std::vector<float> output = nn.forward(inputVec);
        int count = static_cast<int>(std::round(output[0]));
        countMap[num] = count;
    }

    // Print the results
    for (const auto& pair : countMap) {
        std::cout << pair.first << " " << pair.second << std::endl;
    }

    return 0;
}
