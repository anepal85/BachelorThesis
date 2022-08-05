##f_2(x)
def get_x1_x2_y(n,x1=[0,10],x2=[0,6],plot=False):
  l,h = tuple(x1)
  a,b = tuple(x2)
  x1 = np.random.uniform(l,h,n)
  x2 = np.random.uniform(a,b,n)
  response=(30+x1*np.sin(x1))*(4+np.exp(-x2**2))
  if plot:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X,Y = np.meshgrid(np.sort(x1[:100]),np.sort(x2[:100]))
    def res(X,Y):
      return (30+X*np.sin(X))*(4+np.exp(-Y**2))
    ax.plot_surface(X, Y, res(X,Y), rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    plt.show()
  independent = np.column_stack((x1,x2))
  return independent, response

###f_3(x)
def nonlinear_f1(n_sample,d=6,threshold=400):
  X = np.random.uniform(0,4,(n_sample,d))
  vado = np.zeros((d,n_sample))
  for i in range(X.shape[1]):
    vado[i] = X[:,i]**i
  y = np.sum(vado,axis=0)
  p = np.where(y>=threshold,threshold,y)
  return X,p

##f_4(x)
def func_dfclt(n_sample,d=10):
  y = np.zeros((d,n_sample))
  X = np.random.uniform(0,4,(n_sample,d))
  for i in range(X.shape[1]):
    y[i] = (X[:,i]**2-X[:,i])**3 - (2+X[:,i])**2
  y = np.sum(y,axis=0)
  return X, y 
