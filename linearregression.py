from numpy import *

def compute_error_for_line_given_points(b,m,points):
	#intialize it at 0
	totalerror=0
	#for every point
	for i in range(0,len(points)):
		x = points[i,0]
		y= points[i,1]

		#get the difference square it and add it to the total

		totalerror+= (y-((m*x)+b))**2
	#get the average
	return totalerror / float(len(points))

def gradient_descent_runner(points,initial_b,initial_m,learning_rate,number_iterations):
	b = initial_b
	m= initial_m

	#gradient descent

	for i in range(number_iterations):
		#update b  and m with new values by performing the below step to gain the more accurate line

		b,m = step_gradient(b,m, array(points),learning_rate)
	return [b,m]


def step_gradient(b_current,m_current,points, learning_rate):
	b_gradient = 0
	m_gradient= 0
	n = float(len(points))


	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]

		#computing partial derivatives  of our error function
		b_gradient+= -(2/n)* (y-((m_current*x)+b_current))
		m_gradient+= -(2/n)* (x * (y-((m_current*x)+b_current)))

#update  our b and m values using partial derivatives

	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)

	return [new_b,new_m]


	

def run():
	#import data using numpy method
	points = genfromtxt("C:\\Users\\krish\\Desktop\\data1.csv",delimiter=",")

	#step2:  define the hyperparameters

	learning_rate=0.0001 # how fast should the model converge its the same as alpha in the formula. hyper parameters is a balance
	#y = mx+b(slope formula)
	initial_b = 0 
	initial_m = 0
	number_iterations = 1000 #this is the n value in the formula


	#step3 is train the model
	print ("starting gradient descen at b = {0}, m = {1}, error = {2}".format(initial_b,initial_m,compute_error_for_line_given_points(initial_b,initial_m,points)))
	[b,m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate,number_iterations)
	print ("ending gradient descen at b = {0}, m = {1}, error = {2}".format(b,m,compute_error_for_line_given_points(b,m,points)))



if __name__ == '__main__': #meat of the code goes here

	run()
