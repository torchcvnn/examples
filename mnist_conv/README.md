# MNIST classification in the spectral domain

This simple example demonstrates how to code and run a complex valued neural network for classification.

The task does not necessarily make sense but provides complex valued inputs : we classifiy the MNIST digits from their spectral representation.

```bash
python -m pip install -r requirements.txt
python mnist.py
```

An expected output is :

```bash
Logging to ./logs/CMNIST_0
>> Training
100%|██████| 844/844 [00:17<00:00, 48.61it/s]
>> Testing
[Step 0] Train : CE  0.20 Acc  0.94 | Valid : CE  0.08 Acc  0.97 | Test : CE 0.06 Acc  0.98[>> BETTER <<]

>> Training
100%|██████| 844/844 [00:16<00:00, 51.69it/s]
>> Testing
[Step 1] Train : CE  0.06 Acc  0.98 | Valid : CE  0.06 Acc  0.98 | Test : CE 0.05 Acc  0.98[>> BETTER <<]

>> Training
100%|██████| 844/844 [00:15<00:00, 53.47it/s]
>> Testing
[Step 2] Train : CE  0.04 Acc  0.99 | Valid : CE  0.04 Acc  0.99 | Test : CE 0.04 Acc  0.99[>> BETTER <<]
```
