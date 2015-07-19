sentence = []
sentence.append([1,0,0,0,0])
sentence.append([0,1,0,0,0])
sentence.append([0,0,0,0,1])
sentence.append([0,0,1,0,0])
sentence.append([0,0,0,1,0])
sentence.append([0,0,0,0,1])

meaning = []
meaning.append([1,1,0,0,0])
meaning.append([1,1,0,0,0])
meaning.append([1,1,0,0,0])
meaning.append([0,0,1,1,0])
meaning.append([0,0,1,1,0])
meaning.append([0,0,1,1,0])

preds = []

for i in range(trials):
    net_a_p.offset = 1
    # first word
    #ds = SupervisedDataSet(szIn, szIn) 
    #ds.addSample(meaning[0],sentence[0])
    #bp = BackpropTrainer(net_a_p,ds,verbose=False,learningrate=LR)
    #bp.trainEpochs(1)
    #listenMeaning = net_a_p.activate(meaning[0])
    
    #preds.append(listenMeaning)    
    
    for j in range(0,4):
        net_a_p.offset = 1
        # subsequent words
        ds = SupervisedDataSet(szIn, szIn) 
        ds.addSample(meaning[j],sentence[j])
        bp = BackpropTrainer(net_a_p,ds,verbose=False,learningrate=LR)
        bp.trainEpochs(1)
        listenIn = net_a_p.activateOnDataset(ds)
        preds.append(listenIn)
        net_a_p.offset = 1
        #listenMeaning = 3*(listenMeaning + listenIn / (listenMeaning + listenIn))

        #ds = SupervisedDataSet(szIn, szIn) 
        #ds.addSample(listenIn,listenMeaning)
        #bp = BackpropTrainer(net_b_c,ds,verbose=False,learningrate=LR)
        #bp.trainEpochs(1)



