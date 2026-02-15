
# Exercise 1: Create a Deep Learning Model for image classification in PyTorch with CIFAR-10 dataset

## Objective

Develop a model that can classify images from CIFAR-10 dataset

First try a model only with fully connected layers
Create an evaluate.py file that evaluates the model and calculates and stores the evaluation metrics including a confusion matrix

Which are the conclussions?

## Task Formalization

El objetivo de la práctica es realizar la comparativa entre los resultados obtenidos con una red lineal compuesta por Fully Connected y una red Convolucional. En este ejercicio se aborda el desarrollo y el entrenamiento de la red FC para la solución de un problema de clasificación de imágen multiclase.

### Task Formalization (Inference)

La imagen de entrada tiene un tamaño de 3x32x32, al ser RGB. Como se está trabajando con un conjunto de capas FC, es necesario convertir el tensor en un vector de 3072 elementos. Una vez se tienen los datos listos, se alimenta la red de capas Fully Connected, obteniendo a la salida un vector de dimensión 10. 

### Task Formalization (Training)

El entrenamiento de la red será lento, pues una imagen corresponde a un gran número de datos. Cuanto mayor sea el número de capas, más parámetros y pesos deberán ajustarse y más tiempo de procesamiento requerirá. Con Batch Normalization, es posible acelerar la convergencia y lograr mejores resultados en menor tiempo. Se utilizará una técnica dropout para apagar un número determinado de neuronas aleatorias con el fin de evitar overfitting. Como función de pérdida, al tratarse de un ejercicio de clasificación de imágenes, se recurrirá a Entropía Cruzada.

## Evaluation metrics

Para la evaluación se implementa la generación de una matriz de confusión que permita observar de forma gráfica la capacidad de predicción de la red.

## Data Considerations

### Dataset description

Write your answer here

### Data preparation and preprocessing

Write your answer here

### Data augmentation

Write your answer here

## Model Considerations

Write your answer here

### Suitable Loss Functions

Write your answer here

### Selected Loss Function

Write your answer here

### Possible architectures

Write your answer here

### Last layer activation

Write your answer here

### Other Considerations

Write your answer here

## Training

Write your answer here

### Training hyperparameters

Write your answer here

### Loss function graph

![image](../../outs/exercise_03/loss_plot.png)

### Discussion of the training process

Write your answer here

## Evaluation

### Evaluation metrics

Write your answer here

![image](../../outs/exercise_04/train_regression_plot.png)

![image](../../outs/exercise_04/validation_regression_plot.png)

![image](../../outs/exercise_04/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_04/metrics.png)

### Evaluation results

Here you have examples of evaluation results for train, validation and test sets.

Example for train set:

![image](../../outs/exercise_04/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_04/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_04/test_data_points_plot.png)


### Discussion of the results

How the model solves the problem?
Is there overfitting, underfitting or any other issues? 
How can we improve the model?
How this model will generalize to new data?

## Design Feedback loops

Describe the process you have followed to improve the model and the evolution of performance of the model during the process.

You can include a table stating the chanched parameters and the obtained results after the process.


## Questions

Pleaser answer the following questions. Include graphs if necessary. Store the graphs in the `outs/exercise_03` folder.

### Which are the differences you found between previous model and this one?

### Does the model generalizes well to new data?






