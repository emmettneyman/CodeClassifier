Alex Matthys
Emmett Neyman

# Project: Code Classifier


# Project Description:
A machine learning classifier that will tell you what programming language an input string corresponds to. Languages which our classifier can recognize include Java, Python, Haskell, and SQL.


# How to use our project:
Our project consists of a clean, simple web app. Our web app has one page, with a large user input box asking the user to input a string of code. After inputting some code and clicking the button 'Submit', our program will output which language is believes the inputted string corresponds to. Furthermore, our web app has a history display, which keeps track of and displays to the user the past queries (input string and what the result was). 


# How we meet the requirements:
Our project uses Machine Learning to put together a Multi Label Classification project, which takes as input a string and decides (based on a large set of training data we have provided and trained our classifier with) which label to assign to this input string (Java, Python, Haskell, or SQL). If the classifier is unable to come to a decisive conclusion, then it outputs that the code is too generic to be classified. We create a custum class called Classifier() within our classifier.py file, which instantiates an instance of the classifier our program uses. This custom class then has an __init__() function which takes as input the training data and the corresponding training lables, creates a classifier, and trains it with the inputted data. This custom class then has the method predictCode() which takes as input a string, calls .predict on the classifier we created in __init__(). We then have a returnPrediction() method which returns the label that our ML program has assigned to the input string. Finally, we have defined the magic method __str__() in such a way that it returns a string of the past queries run against this Classifier object, allowing us to display a user's previous queries on the web app. 

Our program uses the following three modules: Flask, SKLearn, and Numpy. Finally, in our file frontend.py, we make use of decorators to route our web app to the correct location, and to accept only the specified requests (Post and Get).



## Original Project Proposal (Untouched since the time we submitted this proposal)
For our project, we want to make a program that uses machine learning to tell you the programming language that a given string is in.For example, the user might give our program the string `public static void main(String[] args)` and our program would output `Java`.For our final project, we envision a simple web interface that would allow the user to input their string and would display the result of our classification algorithm. We are a little unsure about what machine learning algorithm to use for this project. We initially thought that K-means clustering would work, but are now unsure since we don't know how we would update the centroids. We then considered supervised algorithms. We understand the k-neighbors concept, and that seems like something we could hard code without using any libraries, but we also started reading about other ways to go about this and came across the Naive Bayes method (although we are unsure exactly how this works). Do you have any advice or suggestions, specifically on how we should go about our ML? Our first goal is to complete the classifier algorithm. After that is complete, we will start working on the web interface, and finally if we have extra time we will work to make the app look pretty. 
