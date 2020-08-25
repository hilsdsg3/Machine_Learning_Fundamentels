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

In the following model diagram, the inputs are the criteria conditions for yes/no. As an example in credit approval,  x1 may represent annual salary, x2 may represent credit length, and x3 may represent a past deliquency. Salary maybe more important than credit length so a factor is used for the inputs. Choosing the right importance or weights is the key to the perceptron algorithm. The weights are represented by the arrows. If these weights are adjusted correctly , the perceptron predicts the binary outcome , "yes" credit approval or "no" disapproval.      

<p float="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram.png" width='350'><img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram2.png" width='350'>
</p>

Mentioned earlier the training data becomes important for the perceptron to use as it modifies importance of the crteria as model weights. Then the perceptron algorithm is used to accurately predict the outcome of new data. This new data contains the characteristic conditions (x values) for approval/disapproval.

</details>

<details>
<summary>--- Overview on the Perceptron Algorthim</summary>
<p>

The following diagram shows the inputs as x variables and the weights, rather the importance of the inputs.
A new concept is the threshold or bias. This is defined as the minimum criteria for a Yes or True result.
The bias is an input also but we don't or can't know the specific conditions for resulting in a yes/no. More description on the bias is folllowing.
</p>

<p align="center"><img width=60% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram_detail3.png"></p>

<p>

Before we dive into the way the algorithm works let use a conceptual example of **Credit approval**.
The input criteria are items such as salary, previous deliquences, and credit history. These inputs are cast x_1,...,x_d. The weights would be the importance of each x criteria. The bias / threshold is the minimum criteria to approve/disapprove credit.

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

Bias = 3 : Minimal criteria for credit approval
```
While training the perceptron seeks to find the optimal weights and the bias term.
If sucessful , an accurate prediction is applied to new data. Then the perceptron has done it's job.

</detail>

<p align="center"><img width=60% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Credit_approval_example.png"></p>

<p align="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/general_perceptron_model_equation.png"/>
</p>

Let's break this model down:
- f(x) represents predicted output of the perception
- The weights (w) multiplied by the critera (x). That result is compared with 0.
- If the result is greater than 0 , f(x) is 1. If not then f(x) is 0.
- If f(x) is 1 then update the weights.

<p align="left">
<img src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/weighted_sum_equation.png.png"/>
</p>
</details>

<details>
<summary>--- Perceptron magic</summary>
<p align="center"><img width=30% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Magic_pic.png"></p>

<p>

The real magic comes when the perceptron updates the weights. The weights are updated in the following equation. There is an error term and that is difference in the true decision and the percieved decision : y - f(x). For each row of training data there are two sets of values : The x values and the result true y values. The y values are true because this is the true outcome of the criteria. That is why it is crucial to do an exploratory data analysis on your training data. If one data point is truely misclassified that one data point is going to skew the final weights.
</p>

<p align="left">
<img src="https://latex.codecogs.com/svg.latex?Update\,\,equation\,\,=:\,w' = w\,+\,error\,*\,x=w +\,(y - f(x))\,x"/>
</p>

<p align="left">
<img src="https://latex.codecogs.com/svg.latex?Error\,\,equation\,\,=:\,error = y\,-\,f(x)"/>
</p>

```
Variables :
y =: real decision yes/no
f(x) =: perceptron/perceived decision
w =: weights
```

<p align="left">
<img src="https://latex.codecogs.com/svg.latex?Perceptron\,\, neurons\;=:output=\left\{\begin{matrix}
-1 & if \, \sum _jw_jx_j\leq threshold\\ 1 & if \, \sum _jw_jx_j> threshold \end{matrix}\right."/>
</p>

<p>
You may ask what about the threshold / bias. I will intergrate that below.
At this point we do not have a good idea for the threshold/bias values but if up the problem differently the bias can be included in with the weights as another variable.
</p>

<p align="center"><img width=60% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram_detail4.png"></p>

<p>

The above diagram has two criteria for simplicity purposes but the diagram adds in the bias term to the inputs.
The x input for the bias is always 1. The weight for the bias may change to whatever the perceptron finalizes. The x component to the bias term is always a 1 because the bias term is either on or off depending on the bias weight.
Including the bias term in the x components will make cleaner coding.  

</details>

<details>
<summary>--- Perceptron learning steps</summary>
<p>

1. Initialize the weights, often randomly or set them with an initial value of 0
```
Training data :
x1    x2  |   y
---- ---- | ----
 1     1  |   1    : first set
 1     0  |   0    : second set
 0     1  |   0    : third set
 0     0  |   0    : fourth set
```

```
Initialize the weights :
w = [0, 0, 0] # [bias, weight, weight]
```

2. For each set of inputs in the set of training examples our perceptron will :
Predict and output , compare it to the expected output , update its weights, if 
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

Let's go through one iteration of updating the weight.

```
If intial w = [0, 0, 0] and bias,x1,x2 = [1, 1, 1] where y=1
then f(x) = 1 if w dot x > 0
= (0 * 1) + (0 * 1) + (0 * 1) = 0 when y=1
Therefore e = y - f(x) = 1
```

```
w_1 + e * 1 = 0 + 1 * 1 = 1 for w_1
w_2 + e * x_1 = 0 + 1 * 1 = 1 for w_2
w_3 + e * x_2 = 0 + 1 * 1 = 1 for w_3
Resulting in w' = [1, 1, 1]
```

This makes sense according to y=1 which is positive. In the above training data, x1 and x2 must be positive together to output a 1 AND since y=1. One important note is bias weight term can be updated by the perceptron but the bias x term is always 1. The perceptron is activaed or postive when the when the bias weight is 1.

Now we have the weights that will predict the first set but we need to do some further work on tuning these weights to have 0 error or successful prediction in all cases.

```

So we can continue with the updated weights : w = [1, 1, 1] but this time
with the 2nd set of data.
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

So in conclusion, after the weights were updated 12 times, we started with a default weight of [0, 0, 0] with the perceptron's assistance we now have a robust set of final weights = [-2, 1, 2]. These weights are correct for all cases of the any training data. Now this example was easily adjusted and used for simplistic terms so one could notice the changes in the weights. 
 </p>

</details>

<details>
<summary>--- More concepts</summary>
<p>

**There are three additional concepts that need explaining.**
</p>

<p>
Epoch is the number of times we’ve iterated through the entire training set. So for the example above, during epoch = 12, we were able to establish weights to classify all of our inputs, but we continued iterating, to be sure that our weights were tried on all of our inputs.
</p>
<p>

Threshold is different than what we now call our bias. Threshold is the maximum number of epoch we will allow to pass while training. There is not built in stopping point of our algorithm. It will continue adding 0 to our weights, on and on, forever. Adding a threshold is one way of stopping our training loop.
</p>

<p>

Learning rate, symbolized by α, is the magnitude at which we increase or decrease our weights during each iteration of training. So a slight modification to the weight update equation.
</p>

<img src="https://latex.codecogs.com/svg.latex?w' = w + {\color{Red} \alpha} \,(y - f(x))\,x"/>

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
