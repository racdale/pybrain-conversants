def buildSRN(i,h,o):
    net = RecurrentNetwork()
    # initialize layers
    net.addInputModule(LinearLayer(i,name='input'))
    net.addModule(SigmoidLayer(h,name='hidden'))
    net.addModule(BiasUnit('bias'))
    net.addOutputModule(SigmoidLayer(o,name='output'))
    net.addModule(LinearLayer(h,name='context'))
    # connect layers
    net.addConnection(FullConnection(net['input'], net['hidden']))
    net.addConnection(FullConnection(net['context'], net['hidden']))
    net.addConnection(FullConnection(net['bias'], net['hidden']))
    net.addConnection(FullConnection(net['bias'], net['output']))
    net.addConnection(FullConnection(net['hidden'], net['output']))
    net.addRecurrentConnection(IdentityConnection(net['hidden'], net['context']))
    return(net)

# get the bare SRNs
net_a_c = buildSRN(szIn,hids,szIn) 
net_a_p = buildSRN(szIn,hids,szIn)
net_b_c = buildSRN(szIn,hids,szIn)
net_b_p = buildSRN(szIn,hids,szIn)

# Chang 'em
#net_a_p.addModule(LinearLayer(hids,name='hidden input from comp'))
#net_a_p.addConnection(FullConnection(net_a_p['hidden input from comp'],net_a_p['hidden'])) # comp > production
#net_b_p.addModule(LinearLayer(hids,name='hidden input from comp'))
#net_b_p.addConnection(FullConnection(net_b_p['hidden input from comp'],net_b_p['hidden'])) # comp > production
# note: listening handled by training function (enforcing linear input from a > b; b > a)
# note: transfer of activation from comp > prod done linearly in training, as well; see trainChangs.py

net_a_c.sortModules()
net_a_p.sortModules()
net_b_c.sortModules()
net_b_p.sortModules()

net_a_c.randomize()
net_a_p.randomize()
net_b_c.randomize()
net_b_p.randomize()
    
#net.sortModules()
#net.randomize()
#net.reset()

                      