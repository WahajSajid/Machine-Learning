import pandas as pd
import numpy  as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0,0],[1,1],[2,2]], [0,1,2])
reg.coef_
print(reg.coef_)