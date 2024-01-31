# Machine Learning Modelling Pipelines in Production

https://www.coursera.org/learn/machine-learning-modeling-pipelines-in-production

## Hyperparameter tuning
**Neural Architecture Search (NAS)**: automating the design of NNs

- Sometimes better than "handmade" models;
- Scalable.

Libraries, such as **Keras Tuning**

How to know if a architecture is optimal:
- Do the model perform well with less hidden units?
- How the model size affects convergence speed?
- Are there any tradeoffs between convergence speed, model size and accuracy?

Code snippet of Keras Tuner to solve MNIST:

```python
!pip install -q -U keras-tuner

import kerastuner as kt

def model_builder (hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))

    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Hyperband is one of the supported strategies, with other such as RandomSearch, BayesianOptimization, Sklearn
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.kear.callbacks.EarlyStopping (monitor='val_loss', patience=5)

tuner.search(x_train, y_train,
             epochs=50, validation_split=.2,
             callbacks=[stop_early])
```

A demonstration file can be seen at [C3_W1_Lab_1_Keras_Tuner](hyperparameter-tuning/C3_W1_Lab_1_Keras_Tuner.ipynb)

## TensorFlow TFX

[TensorFlow TFX](https://www.tensorflow.org/tfx) is a platform to develop production pipelines. The main documentation of its components can be seen [here](https://www.tensorflow.org/tfx/guide/understanding_tfx_pipelines).

A demonstration file can be seen at [C3_W1_Lab_2_TFX_Tuner_and_Trainer](hyperparameter-tuning/C3_W1_Lab_2_TFX_Tuner_and_Trainer.ipynb).

## AutoML

**Macro Architecture Search Space** is a search space where each layer is processed in order.

**Micro Architecture Search Space** is a method where multiple layers are placed together.

Some search strategies:
- Grid search;
- Random Search;
- Bayesian Optimization;
- Evolutionary methods;
- [Reinforcement Learning](references/NEURAL%20ARCHITECTURE%20SEARCH%20WITH%20REINFORCEMENT%20LEARNING.pdf);
- [Progressive NAS](references/PROGRESSIVE%20NEURAL%20ARCHITECTURE%20SEARCH.pdf)

To reduce the cost of precision estimation in the entire sampling space, some methods can be adopted:
- Lower Fidelity Estimates: uses small subsetes or low resolution samples to reduces computational costs in inference of candidates;
- Learning Curve Extrapolation: based on a few iterations, extapolates the performance of candidates;
- Weight Inheritance/Network Morphisms: initialize weights of new architectures based on previously trained architectures.

## AutoML in the cloud

Options:
- Amazon SageMaker Autopilot;
- Microsoft Azure Automated Machine Learning;
- Google Cloud AutoML.