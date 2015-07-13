sentence = []
sentence.append([1,0,0])
sentence.append([0,1,0])
sentence.append([0,0,1])

quiet = []
quiet.append([0,0,1])
quiet.append([0,0,1])
quiet.append([0,0,1])

ds = SupervisedDataSet(2, 2) 
ds.addSample(sentence[0],sentence[1])
bp = BackpropTrainer(net_a_p,ds,verbose=False,learningrate=0.25)






ix = 0
for i in range(100):
    ln = int(rnd()*6)+3
    for j in range(ln):
        inputs.append([1,0])
    for j in range(ln):
        inputs.append([0,1])
outputs = inputs[1:]
inputs = inputs[:-1]
ds = SupervisedDataSet(2, 2) 
for i in range(len(inputs)):
    ds.addSample(inputs[i],outputs[i])