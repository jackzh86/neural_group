===Background===
Neural networks have demonstrated impressive capabilities in handling unstructured data such as text (NLP), video, and audio. However, their effectiveness in processing structured data, particularly for tasks like grouping and counting, remains less explored. This experiment aims to determine whether a neural network can accurately count occurrences of numbers, a fundamental operation in structured data processing.  

===Get Started===
1. Build the Application 
```
g++ -o neural_group NeuralGroupApp-v1.cpp
```
2. Run the Application (Example for Windows)
```
.\neural_group.exe
```

example output:
```
Epoch 1, Total Error: 3.58995
Epoch 101, Total Error: 1.42607
Epoch 201, Total Error: 1.1371
Epoch 301, Total Error: 0.842739
Epoch 401, Total Error: 0.637127
Epoch 501, Total Error: 0.488317
Epoch 601, Total Error: 0.377917
Epoch 701, Total Error: 0.294577
Epoch 801, Total Error: 0.230918
Epoch 901, Total Error: 0.181867
Epoch 1001, Total Error: 0.143766
Epoch 1101, Total Error: 0.113993
Epoch 1201, Total Error: 0.0906123
Epoch 1301, Total Error: 0.0721784
Epoch 1401, Total Error: 0.0576322
Epoch 1501, Total Error: 0.0461048
Epoch 1601, Total Error: 0.0369386
Epoch 1701, Total Error: 0.0296333
Epoch 1801, Total Error: 0.0237999
Epoch 1901, Total Error: 0.0191342
Epoch 2000, Total Error: 0.01543
40 1
30 2
15 1
10 2
20 1
```