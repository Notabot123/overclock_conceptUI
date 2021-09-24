import eel
import numpy as np
from random import randint
from numpy.random import exponential, poisson
from numpy.random import uniform as unf
import json

eel.init("web")

# Exposing the random_python function to javascript
@eel.expose	# not used
def random_python():
	print("Random function running")
	return randint(1,100)

@eel.expose	# not used
def idealised_events():
	print("idealised")
	return randint(1,100)

@eel.expose	
def exprnd(mu=1):
	e = exponential(mu,size=(500,1))
	eel.sleep(0.01)
	return e.tolist()

@eel.expose	
def poissrnd(mu=1):
	e = poisson(mu,size=(500,1))
	eel.sleep(0.01)
	return e.tolist()

@eel.expose	
def unifrnd(mu=1):
	e = unf(0,mu*2,500)
	print(e)
	eel.sleep(0.01)
	return e.tolist()

@eel.expose	
def constrnd(mu=1):
	e = np.tile(mu, (500,1))
	eel.sleep(0.01)
	return e.tolist()



# Start the index.html file
# eel.start("demo copy.html")
eel.start("index copy.html")
