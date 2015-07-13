def buildSRN(i,h,o):
    net = RecurrentNetwork()
    # initialize layers
    net.addInputModule(LinearLayer(i,name='input'))
    net.addModule(SigmoidLayer (h,name='hidden'))
    net.addModule(BiasUnit('bias'))
    net.addOutputModule(SigmoidLayer (o,name='output'))
    net.addModule(LinearLayer(2,name='context'))
    # connect layers
    net.addConnection(FullConnection(net['input'], net['hidden']))
    net.addConnection(FullConnection(net['context'], net['hidden']))
    net.addConnection(FullConnection(net['bias'], net['hidden']))
    net.addConnection(FullConnection(net['bias'], net['output']))
    net.addConnection(FullConnection(net['hidden'], net['output']))
    net.addRecurrentConnection(IdentityConnection(net['hidden'], net['context']))
    return(net)

# get the bare SRNs
net_a_c = buildSRN(3,2,3) 
net_a_p = buildSRN(3,2,3)
net_b_c = buildSRN(3,2,3)
net_b_p = buildSRN(3,2,3)

# Chang 'em
net_a_p.addModule(LinearLayer(2,name='hidden input from comp'))
net_a_p.addConnection(FullConnection(net_a_p['hidden input from comp'],net_a_p['hidden'])) # comp > production
net_b_p.addModule(LinearLayer(2,name='hidden input from comp'))
net_b_p.addConnection(FullConnection(net_b_p['hidden input from comp'],net_b_p['hidden'])) # comp > production
# note: listening handled by training function (enforcing linear input from a > b; b > a)
# note: transfer of activation from comp > prod done linearly in training, as well; see trainChangs.py

net_a_c.sortModules()
net_a_p.sortModules()
net_b_c.sortModules()
net_b_p.sortModules()

   
    
#net.sortModules()
#net.randomize()
#net.reset()

                      