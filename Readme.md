## Fundamental machine learning concepts from various sources

## Concept topics :
- Basic Learning Diagram
- Linear Perceptron algorithm
- Hoeffding's inequality

<details>
  <summary>Basic Learning Diagram</summary>
<p align="center"><img width=40% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Basic_Learning_Problem_Diagram.bmp"></p>
</details>

<details>
  <summary>Linear Perceptron classification algorithm</summary>

#### Linear Perceptron classification algorithm
Purpose : Based on certain criteria, develop an automatic way of classifying as usually a yes/no

Background : Perceptrons are a type of artificial neuron that helps classify the outcome into binary (+1/-1) , True/False, Yes/no. For example a perceptron True/False application is predicting a malignant/benign cancer. Or another application is the approval of credit yes/no.

Model : In the following diagram, the inputs are the specific conditions for a Yes/no condition.
As an example in credit approval. x1 may represent annual salary, x2 may represent credit length, and x3 may represent a past deliquency. Salary maybe more important than credit length so a factor is used for the inputs.
Choosing the right factor or weights is the key to the perceptron algorithm. The weights are represented by the arrows. If these weights are adjusted correctly , the perceptron predicts the binary outcome , "yes" crdit approval or "no" disapproval.      
<p align="center"><img width=40% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram.bmp"></p>

#### Main equation : Linear formula
Formula Definition : The perceptron algorithm finds a yes/no within training data to predict the outcome of new data. This new data contains the characteristic conditions (x values) for approval/disapproval. After the perceptron finds the final weights, these are used in conjuction with the charteristics (x values) to obtain an accurate prediction.

Following the perceptron neuron equation is described as :
- output is the output of our formula, which is called the activation of our perceptron.
- The if branches start with the same ∑ summation that takes the inputs (x) and multiplies the weight (w) and then adds them all. This weighted sum formula is in the following and is represented by dot product notation.
If the weighted sum is less than or equal to our threshold, or bias, b, then our output will be 0 or -1 (disapproval)
If the weighted sum is greater than our threshold, or bias, b, then our output will be 1 (approval)
- The resulting inequailty is described by the Heaviside Step function.   
<img src="https://latex.codecogs.com/svg.latex?weighted\, sum\,=: x_1w_1+x_2w_2+x_3w_3....x_dw_d \, =\, w_T \cdot x_j"/>

<img src="https://latex.codecogs.com/svg.latex?Perceptron\,\, neurons\;=:output=\left\{\begin{matrix}
-1 & if \, \sum _jw_jx_j\leq threshold\\ 1 & if \, \sum _jw_jx_j> threshold \end{matrix}\right."/>

<img src="https://latex.codecogs.com/svg.latex?General\, equation =: \;{\color{Red}h}(x)=sign((\sum_{i=1}^{d}{\color{Red}w_i}x_x)-{\color{Red}threshold})"/>

<img src="https://latex.codecogs.com/svg.latex?Heaviside\, Step\, function \, =: f(x) = {\;x\;<=\;b:-1,x>b:1}"/>
<p align="center"><img width=40% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Heaviside_step_function.bmp"></p>

<img src="https://latex.codecogs.com/svg.latex?Alternative\;to\;perceptron\,model\;=:f(x)=\left\{\begin{matrix}
1 & if \, w\cdot x +  > 0\\ 0 &\, otherwise \end{matrix}\right."/>
An alternative perceptron model is more commonly displayed as the summation is represented by dot product notation.
The threshold becomes the bias (b) term. The result of the sum plus the bias are compared to 0. Therefore a large bias terms will certainly "activate" or result in a 1 and be classified as a positive/+1/True/Yes value.  

<p align="center"><img width=40% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Perceptron_diagram_detail.bmp"></p>

<img src="https://latex.codecogs.com/svg.latex?weighted\, sum\,=: x_1w_1+x_2w_2+x_3w_3....x_dw_d \, =\, w_T \cdot x_j"/>

#### Credit approval example :
Continuing the example with the above equation, x_1,...,x_d represent attributes (dimensions) of a credit approval. The approval/disapproval is indicated by a h(x) in the general equation. This binary output is positive or negative for simplicity purposes of the above formula. The threshold of the above equation is important and is in this case the minimum characteristics to approve/disapprove credit.

Describing the problem further : 
- Decsion : Approval/disapproval of credit
- Inputs :
x1 = 1 : Salary
x2 = 0 : Previous deliquencies
x3 = 1 : Credit history
- Weights :
w1 = 4 : Salary , most significant
w2 = 3 : Previous deliquencies
w3 = 2 : Credit history , least significant
- Bias : 3

<p align="center"><img width=40% src="https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Credit_example.bmp"></p>

<img src="https://latex.codecogs.com/svg.latex?(1*4)\, + \,(0*3)\, + (1*2) = 6"/>
<img src="https://latex.codecogs.com/svg.latex?6 > 3 =: 1 so \,\,the \,\,result\,\, is\,\, a \,\,1\,\, for\,\, approval"/>

#### Perceptron learning steps
1. Initialize the weights, often randomly or set them with an initial value of 0
'''
Training data :
x1    x2  |   y
---- ---- | ----
 1     1  |   1
 1     0  |   0
 0     1  |   0
 0     0  |   0
 '''
'''
Initialize the weights :
w = [0, 0, 0] # [bias, weight, weight]


 '''

2. For each set of inputs in the set of training examples our perceptron will :
Predict and outputt , compare it to the expected output , update its weights, if 
the expected output does not equal the actual output and move to the next set of inputs.
Further concepts : 
- We can define how well the perceptron is performing on the known training set data by the error (e).
The goal for the perceptron have as minimal error as possible or 0. 
''' 
e = expected output - actual output = y - f(x)
'''
- Adjustment to the weights of the perceptron. While the input value (x) cannot be adjusted the weights can as w'.
'''
if the error is +1 then the weight must be adjusted to   w' = w + x
if the error is -1 then the weight must be adjusted to   w' = w - x
if the error is 0 then the perceptron correctly predicted the expected output   w' = w 
'''
To further clarify the equations :
'''
w' = w + (error) * x
'''








#### Main steps of the algorithm :

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
