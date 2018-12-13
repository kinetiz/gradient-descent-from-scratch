import numpy as np
import matplotlib.pyplot as plt
import time

# gen x,y data
data = np.random.multivariate_normal([1,1], [[1,2],[2,1]],size = 100)

# plot data points 
X= [i[0] for i in data]
Y= [i[1] for i in data]
plt.scatter(X,Y)
plt.show()

# --- modeling

# === objective function
# yh = m*x + b #-> find m and b that best estimate y. Thus, minimize error between yh and y
# err = (y-yh)**2

# ================ Initialise parameters =============================
m = 0
b = 0
iteration = 100
learning_rate = 0.1
# ====================================================================
m_lst = []
b_lst = []
sqerr_lst = []
grad_m_lst = []
grad_b_lst = []
N=100
# update weights (m&b) from each iteration over all data points (batch gradient descent)
for ite in range(iteration):
    # 1. calculate gradients
    grad_m, grad_b  = 0, 0
    for x, y in zip(X,Y):
        grad_m += -(2/N)*x*(y - ((m*x) + b)) 
        grad_b += -(2/N)*(y - ((m*x) + b))
    grad_m_lst.append(grad_m)
    grad_b_lst.append(grad_b)
    # 2. update weights
    m -= grad_m*learning_rate
    b -= grad_b*learning_rate
    m_lst.append(m)
    b_lst.append(b)
    print('--- iter: {0}'.format(ite))
    print('m: {0}'.format(m))
    print('b: {0}'.format(b))
    
    # 3. generate model
    # yh = m*x + b #-> find m and b that best estimate y. Thus, minimize error between yh and y
    Yh = [m*x + b for x, y in zip(X,Y) ]
    
    # err = (y-yh)**2
    sqerr= [ (y-yh)**2 for y,yh in zip(Y, Yh)]
    sum_sqerr = np.sum(sqerr)/N
    sqerr_lst.append(sum_sqerr)
    print('sum square error: {0}'.format(sum_sqerr))
    print('improved error : {0}'.format(sqerr_lst[len(sqerr_lst)-2] - sum_sqerr))
    time.sleep(0.2)
    
# print final model
print('m:{0}, b:{1}'.format(m, b))
# ========= plot evoluation
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title('Weight change over iterations ')
ax1.set_xlabel('m')
ax1.set_ylabel('b')

ax2.set_title('Error changes over iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MSE')

ax1.plot(m_lst, b_lst, 'r--' ,m_lst[-1],b_lst[-1],'gx')
ax2.plot(sqerr_lst, 'r-')

plt.show()


# ========= plot compare with linear model from theory
# theory linear model
fit = np.polyfit(X,Y,1)
fit_fn = np.poly1d(fit) 
# fitted linear model from gradient descent
gd_fn = np.poly1d([m, b]) 

# fit_fn is now a function which takes in x and returns an estimate for y
fig2 = plt.figure('Fitting')
ax3 = fig2.add_subplot(111)
ax3.set_title('Fitted models')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
#ax3.plot(X,Y, 'yo', X, fit_fn(X), '--k', X, gd_fn(X), '--r')
ax3.plot(X,Y, 'yo', label='Data')
ax3.plot(X, fit_fn(X), '--k', label='Theory linear model: {0}'.format(fit))
ax3.plot(X, gd_fn(X), '--r', label='GD linear model: {0}'.format([m, b]))
ax3.legend()
fig2.show()
