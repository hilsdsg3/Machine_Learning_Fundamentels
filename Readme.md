## Fundamental machine learning concepts

## Concept topics :
- Basic Learning Diagram
- Linear Perceptron algorithm
- Hoeffding's inequality

<details>
  <p>

  <summary>Basic Learning Diagram</summary>
  </p>

<p align="left"><img width=60% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Basic_Learning_Problem_Diagram.png"></p>
</details>

<details>
  <summary>Linear Perceptron classification algorithm </summary>
<details>
<summary>--- Background</summary>
<p>

One basic machine learning classification algorithim model is called the Linear Perceptron. The Linear Perceptron model attempts to classify values and result in a final linear equation. Some examples are basic credit approval and disease detection (malignent/benign). The perceptron uses certain criteria as x inputs and automatically adjusts the importance of the criteria based on traning data. Then the resulting perceptron algorithm is used to predict based on new instances of the same criteria for a binary result of yes/no.
</p>

```
Some examples
+1/-1 - Sensor signals
1/0 - Bits (computer)
True/False - Disease detection (cancer or not), fraud detection
Yes/No - Credit approval
```

On the left the model diagram, the inputs are denoted with an x and they are defined as the criteria conditions for yes/no. As an example is credit approval,  x1 may represent annual salary, x2 may represent credit length, and x3 may represent past deliquencies. Salary maybe more important than credit length so a factor is used for the inputs. Choosing the right importance or weights is the key to the perceptron algorithm. The weights are represented by the arrows. If these weights are adjusted correctly , the perceptron predicts the binary outcome , "yes" credit approval or "no" disapproval.      

<p float="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram.png" width='300'><img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram2.png" width='300'>
</p>

To make the algorithm effective, training data becomes important to use as the algorithm modifies the importance or weights the characteristic conditions. Then the perceptron algorithm is used to accurately predict the outcome of new data. This new data contains the characteristic conditions and with the final percepton algorithm the result is an accurate approval/disapproval.

</details>

<details>
<summary>--- Perceptron Algorthim Overview</summary>

<p>

The general model equation evaluates each training set of data and classifies them result as a +1 or 0. These classifications can easily be changed to a +1/-1 instead of 1/0 but they must always be binary , either one or the other.  

<p align="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/general_perceptron_model_equation.svg"/>
</p>

<p>
```
Let's break this model down further:
- f(x) represents predicted output of the perception
- The weights (w) multiplied by the critera (x). That result is compared with 0.
- If the result is greater than 0 , f(x) is 1. If not then f(x) is 0.
- If f(x) is 1 then update the weights.
```
</p>

<p>

The following diagram shows the inputs as x variables and the weights or the importance of the inputs.
A new concept that is shown is the threshold or bias. This threshold is defined as the minimum criteria for a Yes or True result. All of these weights and the threshold is given an initial number for the perceptron to modify. The threshold initial number does not have to be correct because the perceptron will automatically converge on a final value.
</p>

<p align="center"><img width=60% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram_detail3.png"></p>

<p>

Let use a conceptual example of **Credit approval**. The input criteria (x) are items such as salary, previous deliquences, and credit history. These inputs are cast x_1,...,x_d. The weights would be the importance of each x criteria. The bias / threshold is the minimum criteria to approve/disapprove credit.

</p>

```
Inputs :
x1 = 1 : Salary
x2 = 0 : Previous deliquencies
x3 = 1 : Credit history

Weights :
w1 = 4 : Salary , most significant
w2 = 3 : Previous deliquencies
w3 = 2 : Credit history , least significant

Threshold = 3 : Minimal criteria for credit approval
```
The key to most algorithms including this one is the automatic adjustments of the weights and the threshold. 
The perceptron adjusts the weights based off of training data. This training data is pre-determined to be True/False.
If the perceptron finds a classification or a line that separates the training data into True/False, we have met our goal. Then an accurate prediction is applied to new data for classifying it into True/False using the new classification line.

</detail>

<p align="center"><img width=60% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Credit_approval_example.png"></p>

<p align="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/weighted_sum_equation.svg"/>
</p>
</details>

<details>
<summary>--- Perceptron magic</summary>
<p align="center"><img width=30% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Magic_pic.png"></p>

<p>

The key to the perceptron is updating the weights in the following equation to obtain the accurate classification line. The error term is the difference in the true decision and the percieved decision : y - f(x). For each set of training data there are two sets of values : The x values and the result true y values. The y values are true because this is the true outcome of the criteria. That is why it is crucial to do an exploratory data analysis on your training data. If one data point is truely misclassified that one data point is going to skew the final weights.
</p>

<p float="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/linear_perceptron_update_equation.svg" width='300'>
</p>

<p float="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/linear_perceptron_error_equation.svg" width='250'>
</p>

```
Variables :
y = Accurate classification of the training data set, real decision , yes/no
f(x) = perceptron/perceived classification of each set of training data
w = original weights
w' = weight update
error = Difference of the Accurate classification and perceptron classification 
```
<p>
Thre following diagram makes programming easier. The criteria inputs are x1 and x2. Although, now we have moved the threshold into the inputs as a 1. Remember this threashold was the MIN for a True/Yes condition. By moving the threshold to the inputs, the flexibility in adjusting the threshold weight is clear and done by the perceptron. The threshold term (x*w) may indeed be a 0 because of the weight (1*0=0). But let the perceptron determine that from the training data.
</p>

<p align="center"><img width=60% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram_detail4.png"></p>

  

</details>

<details>
<summary>--- Perceptron learning steps</summary>
<p>

- The goal is to correctly predict the y column with zero error.

- Initialize the weights, often randomly or set them with an initial value of 0
```
Initialize the weights :
w = [0, 0, 0] # [bias, weight, weight]

We are given a training set with three different input (threshold-x0,x1,x2)
Training data :
 x0   x1   x2  |   y
---- ---- ---- | ----
 1    1     1  |   1    : first set
 1    1     0  |   0    : second set
 1    0     1  |   0    : third set
 1    0     0  |   0    : fourth set
```
- For each set of inputs in the set of training examples our perceptron will :
Predict and output a 1/0, compare it to the expected output , update its weights, if 
the expected output does not equal the actual output and move to the next set of inputs.
Further concepts : 
- We can define how well the perceptron is performing on the known training set data by the error (e).
The goal for the perceptron have as minimal error as possible or 0. 
```
e = expected output - actual output = y - f(x)
```
- Adjustment to the weights of the perceptron. While the input value (x) cannot be adjusted, the weights can be adjusted and set to w' by adding or subtracting x.

```
if the error is +1 then the weight must be adjusted to   w' = w + (1 * x) = w + x
if the error is -1 then the weight must be adjusted to   w' = w + (-1 * x) = w - x
if the error is 0 then the perceptron correctly predicted the expected output   w' = w 
```

As the perceptron cycles through the training data rows, a w' results.

```
w' = w + (error) * x
```

Let's go through one iteration of updating the weights.

```
1. Intialize the weights
If intial w = [0, 0, 0] and bias,x1,x2 = [1, 1, 1] where y=1
then f(x) = 1 if w dot x > 0

2. Apply the weighted sum equation
= (0 * 1) + (0 * 1) + (0 * 1) = 0 when y=1
Therefore e = y - f(x) = 1

3. Apply the update w' equation
w_1 + e * 1 = 0 + 1 * 1 = 1 for w_1
w_2 + e * x_1 = 0 + 1 * 1 = 1 for w_2
w_3 + e * x_2 = 0 + 1 * 1 = 1 for w_3
Resulting in w' = [1, 1, 1]
```

Now we have the weights that will predict the first set but we need to do some further work on tuning these weights to have 0 error or successful prediction in all cases.

```
Starting with these weights [1, 1, 1] but this time
with the 2nd set of data istead of the 1st set.
w    = [1, 1, 1] # updated weights
x    = [1, 1, 0] # 2nd set of data
y    = 0
w * x = (1 * 1) + (1 * 1) + (1 * 0) > 0 so
f(x) = 1
e    = -1
w'   = w + -1x = [0, 0, 1]
'---------------------------------
w    = [0, 0, 1] # updated weights
x    = [1, 0, 1] # 3rd set of data
y    = 0
f(x) = 1
e    = -1
w    = w + -1x = [-1, 0, 0]
'---------------------------------
w    = [-1, 0, 0] # updated weights
x    = [1, 0, 0] # 4th set of data
y    = 0
f(x) = 0
e    = 0
w    = w + 0x = [-1, 0, 0]
'---------------------------------
w    = [-1, 0, 0] # updated weights
x    = [1, 1, 1] # restart at the 1st set again
y    = 1
f(x) = 0
e    = 1
w    = w + 1x = [0, 1, 1]
'---------------------------------
w    = [0, 1, 1] # updated weights
x    = [1, 1, 0] # the 2nd set again
y    = 0
f(x) = 1
e    = -1
w    = w + -1x = [-1, 0, 1]
'---------------------------------
w    = [-1, 0, 1] # updated weights
x    = [1, 0, 1]  # the 3rd set again
y    = 0
f(x) = 0
e    = 0
w    = w + 0x = [-1, 0, 1]
'---------------------------------
w    = [-1, 0, 1] # updated weights
x    = [1, 0, 0]  # the 4rd set again
y    = 0
f(x) = 0
e    = 0
w    = w + 0x = [-1, 0, 1]
'---------------------------------
w    = [-1, 0, 1] # updated weights
x    = [1, 1, 1] # 1st set
y    = 1
f(x) = 0
e    = 1
w    = w + 1x = [0, 1, 2]
'---------------------------------
w    = [0, 1, 2] # updated weights
x    = [1, 1, 0] # 2nd set
y    = 0
f(x) = 1
e    = -1
w    = w + -1x = [-1, 0, 2]
'---------------------------------
w    = [-1, 0, 2] # updated weights
x    = [1, 0, 1] # 3rd set
y    = 0
f(x) = 1
e    = -1
w    = w + -1x = [-2, 0, 1]
'---------------------------------
w    = [-2, 0, 1] # updated weights
x    = [1, 0, 0] # 4th set
y    = 0
f(x) = 0
e    = 0
w    = w + 0x = [-2, 0, 1]
'---------------------------------
w    = [-2, 0, 1] # updated weights
x    = [1, 1, 1] # 1st set
y    = 1
f(x) = 0
e    = 1
w    = w + 1x = [-1, 1, 2]
'---------------------------------
w    = [-1, 1, 2] # updated weights
x    = [1, 1, 0] # 2nd set
y    = 0
f(x) = 0
e    = 0
w    = w + 0x = [-1, 1, 2]
'---------------------------------
w    = [-1, 1, 2] # updated weights
x    = [1, 0, 1] # 3rd set
y    = 0
f(x) = 1
e    = -1
w    = w + -1x = [-2, 1, 1]
'---------------------------------
w    = [-2, 1, 1] # updated weights
x    = [1, 0, 0]  # 4th set
y    = 0
f(x) = 0
e    = 0
w    = w + 0x = [-2, 1, 1]
'---------------------------------
w    = [-2, 1, 1] # updated weights
x    = [1, 1, 1] # 1st set
y    = 1
f(x) = 0
e    = 1
w    = w + 1x = [-1, 2, 2]
'---------------------------------
w    = [-1, 2, 2] # updated weights
x    = [1, 1, 0] # 2nd set
y    = 0
f(x) = 1
e    = -1
w    = w + -1x = [-2, 1, 2]
'---------------------------------
w    = [-2, 1, 2] # updated weights
x    = [1, 0, 1] # 3rd set
y    = 0
f(x) = 0
e    = 0                          # No Error!
w    = w + 0x = [-2, 1, 2]
'---------------------------------
w    = [-2, 1, 2] # updated weights
x    = [1, 0, 0] # 4th set
y    = 0
f(x) = 0
e    = 0                          # No Error!
w    = w + 0x = [-2, 1, 2]
'---------------------------------
w    = [-2, 1, 2] # updated weights
x    = [1, 1, 1] # 1st set
y    = 1
f(x) = 1
e    = 0                          # No Error!
w    = w + 0x = [-2, 1, 2]
'---------------------------------
w    = [-2, 1, 2] # updated weights
x    = [1, 1, 0]  # 2nd set
y    = 0
f(x) = 0
e    = 0                          # No Error!
w    = w + 0x = [-2, 1, 2]
'---------------------------------
w    = [-2, 1, 2] # updated weights
x    = [1, 0, 1] # 3rd set
y    = 0
f(x) = 0
e    = 0                          # No Error!
w    = w + 0x = [-2, 1, 2]
```

<p>

So in conclusion, after the weights were updated 12 times, we arrive at the final weight [-2, 1, 2]. We started with a default weight of [0, 0, 0] with the perceptron's assistance we now have a robust set of final weights. These weights are correct for all cases of the any training data. Now this example was easily adjusted and used for simplistic terms so one could notice the changes in the weights. 
 </p>

</details>

<details>
<summary>--- Additional  concepts</summary>
<p>

**There are three additional concepts that need explaining.**
</p>

<p>
Epoch is the number of times we’ve iterated through the entire training set. So for the example above, during we only ran the training data two times through. In the real world we have several input sets so it is necessary to have at least as many Epochs for the perceptron to converge on a solution.
</p>
<p>

Threshold is different than what we now call our bias. Threshold is the maximum number of epochs we will allow to pass while training. There is not a built in stopping point of our algorithm. Adding a threshold is one way of stopping our training loop.
</p>

<p>

Learning rate, symbolized by α, is the magnitude at which we increase or decrease our weights during each iteration of training. So the alpha is multipled by the error an the x as the following equation. The manual example above does not demostrate this too clearly but the learning rate number is flexible so that if you choose 0.2 or 0.3 , the perceptron is going to converge. Generally, set the learning rate to 20% of the x range.
</p>

<p float="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/linear_perceptron_updated_update_equation.svg" width='250'>
</p>

</details>

<details>
<summary>--- Programming steps to the Perceptron</summary>

1. The dimensions are the characteristics (x1,x2,x3...,xd).
Assign random number(s) as weights according to the amount of characteristics given the training set x (x1, y1),(x2, y2),...(xf,yf)
<img src="https://latex.codecogs.com/svg.latex?{\color{Red}h}(x)=sign({\color{Red}w_T}x)"/>
2. Obtain the +/- sign of h(x) for all the x points multiplied by their weight.
3. Assign a learning rate constant, nu. The learning rate is the update factor value that is multiplied by the weights.
It is best to choose a learning constant that is a fraction (~20%) of the range in the x values.   
4. Compare the h(x) sign with the respective y sign. Denote the misclassified h(x).  
5. Pick at random a misclassified h(x) :
<img src="https://latex.codecogs.com/svg.latex?sign({\color{Red}w_T}x)\neq{y_n}"/>
6. Use the learning rate (nu), original weight (w) and the correct y sign to get the new weight  
<img src="https://latex.codecogs.com/svg.latex?{{w_T}^{'}}=w_T+\nu y_n"/>
7. For number of misclassified points, repeat step 6.
8. If linearly separable points or in other words if the points can be bisected/divides correctly by a line, this is called convergence.
9. Plot the resulting line with the points.
</details>
</details>


<details>
  <summary>Hoeffding's inequality</summary>
  <img src="https://latex.codecogs.com/svg.latex?\mathbb{P}\left [ \left | E_{in}(h)-E_{out}(h) > \epsilon \right | \right ]\leq 2e^{-2\epsilon ^2N}"/>
</details>

<details>
  <summary>References :</summary>
* [Fundamentals of Machine Learning - Caltech CS156 taught by Dr. Abu-Mostofa](https://work.caltech.edu/telecourse)
* [Machine Learning - Coursera - Andrew Ng](https://www.coursera.org/learn/machine-learning/home/welcome)
* [Reinforcement Learning - UC Berkeley - Sergey Levine](https://www.youtube.com/watch?v=SinprXg2hUA&list=PLkFD6_40KJIwhWJpGazJ9VSj9CFMkb79A&index=1)
</details>
