import eel
import numpy as np
from random import randint
from numpy.random import exponential, poisson
from numpy.random import uniform as unf
import pandas as pd
import sys
from classes.parse_drawflow import parse_chart


eel.init("web")

# Exposing the random_python function to javascript
@eel.expose	# just example for sim result return
def random_python():
	print("Random function running")
	return randint(1,100)

# Exposing the random_python function to javascript
@eel.expose	
def js_to_py(data_json):
	""" Retrieve diagram from drawflow """
	print("Diagram looks like this in JSON you see Python...")
	# print(data_json)
	parse_chart(data_json)
	# return data_json

@eel.expose	
def read_table(data):
	d = data.splitlines()
	d = [i.split(',') for i in d]
	df = pd.DataFrame(data=d)
	h = df.to_html()
	return h

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
	eel.sleep(0.01)
	return e.tolist()

@eel.expose	
def constrnd(mu=1):
	e = np.tile(mu, (500,1))
	eel.sleep(0.01)
	return e.tolist()

# Start the index.html file
eel.start("index.html")
