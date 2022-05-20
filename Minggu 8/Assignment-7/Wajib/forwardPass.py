def forwardPass(inputs,weight,bias):
    w_sum = np.dot(inputs,weight) + bias 

    act = w_sum
    return act