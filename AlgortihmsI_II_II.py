### algo-1 Sample Complexity but with MAPE as Accuracy 
@timethis
def sample_complexity_with_MAPE(func,required_acc=95,initial_sample_size=1000,test_samples=5000):
  ##Normalization 
  scalar = MinMaxScaler()
  x_test,y_test = func(test_samples)
  mape = []
  samples = []
  evaluated_acc = 0 
  x_train, y_train = func(initial_sample_size)

  while evaluated_acc<=required_acc:
    y_train_scaled = scalar.fit_transform(y_train.reshape(-1,1))
    model = train(x_train, y_train_scaled)
    pred = model.predict(x_test)
    pred_inv = scalar.inverse_transform(pred)
    
    mean_ape = mean_absolute_percentage_error(y_test,pred_inv.flatten())*100
    mean_ap = np.round(mean_ape,3)
    evaluated_acc = 100-mean_ap
    
    print(f'&{initial_sample_size}&{np.round(np.sqrt(mean_squared_error(pred_inv, y_test)),3)}&{np.round(r2_score(pred_inv, y_test),3)}&{np.round(mean_ap,3)}\\\\')
    samples.append(initial_sample_size)

    initial_sample_size = 2*initial_sample_size
    additional_x_train, additional_y_train = func(int(initial_sample_size/2))

    #concatenate with the previous ones 
    x_train, y_train = np.concatenate((x_train, additional_x_train),axis=0), np.concatenate((y_train, additional_y_train),axis=0)
    mape.append(mean_ap)
  return initial_sample_size/2,mape,samples 

## Sample Complexity Improved 
##algo 1.2 
## Input : depth and N_high from sample complexity  
##Retruns optimal sample size 
@timethis
def better_sc(func,depth,N_high,required_acc=95,test_samples=5000):
  scalar = MinMaxScaler()
  x_test,y_test = func(test_samples)
  target_mean = np.mean(y_test)  
  X_train, Y_train = func(N_high)
  Y_train = scalar.fit_transform(Y_train.reshape(-1,1))

  N_low = N_high/2 
  for i in range(depth):
    N = int((N_high+N_low)/2)
    x_train, y_train = X_train[:N], Y_train[:N]
    model = train(x_train, y_train)
    pred = model.predict(x_test)

    pred_inv = scalar.inverse_transform(pred)   
    RMSE = np.sqrt(mean_squared_error(y_test,pred_inv.flatten()))
    target_average_error = (RMSE/target_mean)*100
    target_average_accuracy = 100-target_average_error

    mean_ape = tf.keras.metrics.mean_absolute_percentage_error(y_test,pred_inv.flatten()).numpy()
    print(f'&{N}&{np.round(RMSE,3)}&{np.round(r2_score(pred_inv, y_test),3)}&{np.round(mean_ape,3)}&{np.round(target_average_error,3)}\\\\')
    if target_average_accuracy<required_acc:
      N_low = N
    else:
      N_high = N 
  return N_high

##algo 2 
##iterative approach with patiece 
## Input variable are configureable 
## If high performance is required use higher value for patience but requires larger amount of time to compute
## Return list of evaluation metric with optimal sample size
@timethis
def iterative_with_patience(func,test_sample_size=5000,initial_sample_size=1000, min_number_of_training = 5,patience = 5,step_size=1000):
  scalar = MinMaxScaler()
  
  x_test,y_test = func(test_sample_size)
  target_mean = np.mean(y_test)

  sample = []
  rmse_list = [] 
  globalmin_list = [] ##for calculating global minimum

  x_train, y_train = func(initial_sample_size)

  while True:
    y_train_s = scalar.fit_transform(y_train.reshape(-1,1))
    model = train(x_train, y_train_s)
    pred = model.predict(x_test)
    pred_inv = scalar.inverse_transform(pred)

    RMSE = np.sqrt(mean_squared_error(y_test,pred_inv.flatten()))
    target_average_error = (RMSE/target_mean)*100 ### Normalised RMSE in percentage 
    target_average_accuracy = 100-target_average_error 
    rmse_list.append(target_average_error)
    sample.append(initial_sample_size)
    
    ### Other error metrics 
    mean_ape = mean_absolute_percentage_error(y_test,pred_inv.flatten())*100

    initial_sample_size = initial_sample_size+step_size

    additional_x_train, additional_y_train = func(step_size)

    #concatenate with the previous ones 
    x_train, y_train = np.concatenate((x_train, additional_x_train),axis=0), np.concatenate((y_train, additional_y_train),axis=0)

    print(f'&{initial_sample_size}&{np.round(target_average_error,3)}&{np.round(r2_score(pred_inv, y_test),3)}&{np.round(mean_ape,3)}\\\\')
    
    if len(rmse_list)>min_number_of_training:
      minimum = min(rmse_list)
      if len(globalmin_list):
        if minimum<globalmin_list[-1]:
          globalmin_list = []
      
      globalmin_list.append(minimum)

      if len(globalmin_list)>patience:
        return rmse_list, sample[-patience]
    
  return rmse_list, initial_sample_size
