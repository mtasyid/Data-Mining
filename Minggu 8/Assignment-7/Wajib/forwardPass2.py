def forwardPass(inputs,weight,bias,activation = 'linear'):
    w_sum = np.dot(inputs,weight) + bias

    if activation is 'relu' :
        act = np.maximum(w_sum,0)
    else :
        act = w_sum
    
    return act