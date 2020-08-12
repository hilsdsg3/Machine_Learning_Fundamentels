## This readme is a few fundamental machine learning concepts.
This serves as notes from various instructions that I have received.

## Concept topics :
- Basic Learning Diagram
- Linear Perceptron algorithm
- Hoeffding's inequality

<details>
  <summary>Basic Learning Diagram</summary>
<p align="center"><img width=40% src="https://https://github.com/hilsdsg3/Machine_Learning_Fundamentels/blob/master/meta_data/Basic_Learning_Problem_Diagram.bmp"></p>
</details>
https://github.com/hilsdsg3/Econometric_data/blob/master/meta_data/media/Bull_bear.jpg
#### Linear Perceptron classification algorithm
Purpose : Based on certain criteria, develop an automatic way of classifying as usually a yes/no
Broad Formula Definition : An algorithm that finds a trend within training data.  
Applications :
To approve/disapprove someone for credit based the given set of data.
This data is has already been classified as approve/disapprove and will be used as the training data for the trendline  
For example the salary and previous deliquencies are the charateristics .... x1, x2. Also with this training set you have the approve/disapprove data as "y". Y is usually binary as in 1 (approve) or 0 disapprove.
#### Main equation : Linear formula
<img src="https://latex.codecogs.com/svg.latex?{\color{Red}h}(x)=sign((\sum_{i=1}^{d}{\color{Red}w_i}x_x)-{\color{Red}threshold})"/>

#### Example :
For x = (x_1,...,x_d) attributes (dimensions) of a customer,
approve/disapprove if h(x) is positive/negative.
The threshold of the above equation is important and is in this case the minimum charateristics to approve/disapprove credit.

#### Main steps of the algorithm :

1. The dimensions are the characteristics (x1,x2,x3...,xd). So assign a random number as weights according to the amount of charateristics given the training set (x1, y1),(x2, y2)
<img src="https://latex.codecogs.com/svg.latex?{\color{Red}h}(x)=sign({\color{Red}w_T}x)"/>
2. Obtain the sign of h(x) for all the points.

3. Pick at random a misclassified x point where the sign with the weights
does not agree with the y:
<img src="https://latex.codecogs.com/svg.latex?sign({\color{Red}w_T}x)\neq{y_n}"/>
