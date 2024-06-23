import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import matplotlib.pyplot as plt 
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plt_graph(A):
    pi_range = np.arange(0,2*np.pi, 2*np.pi/A.shape[0])
    x_plt = np.cos(pi_range)
    y_plt = np.sin(pi_range)
    plt.scatter(x_plt, y_plt)
    for i in range(0,A.shape[0]):
        for j in range(i,A.shape[1]):
            if (A[i,j]):
                plt.plot([x_plt[i], x_plt[j]], [y_plt[i], y_plt[j]], alpha = (A[i,j]/A.max()).item())
    plt.axis('square')
    plt.axis([-1,1,-1,1])
    plt.show()
    
def sp_roll(A, roll_n = 1):
    A = sp.hstack((A[:, roll_n:], A[:, :roll_n]), format='csr')
    A = sp.vstack((A[roll_n:, :], A[:roll_n, :]), format='csr')
    return A

def torch_roll(A, roll_n = 1):
    A = torch.roll(A,-roll_n,0)
    A = torch.roll(A,-roll_n,1)
    return A

# probably don't use due to floating vertices
def graph_resize(A, N, connected = None):
    A_len = A.shape[0]
    assert A_len < N
    assert connected in [None, 'all', 'one'] # i think None is the intended way
    
    A._shape = (N,N)
    A.indptr = np.hstack((A.indptr,[A.indptr[-1]] * (N - A_len)))
    if connected is not None:
        if connected == 'all': # the newly added vertices are connected to all vertices
            A[A_len:,:N-1] = 1/N
            A[:N-1,A_len:] = 1/N
        if connected == 'one': # they only connect to one other vertice and hope they are kinda even
            for i in range(N - A_len):
                A[i, i + A_len] = 1/N
                A[i + A_len, i] = 1/N
    return A

# probably don't use due to floating vertices
def gen_graph(N, prob, scale=1, step=1):
    vals = torch.rand(int(N*(N+1)/2)) # values
    A = torch.zeros(N, N)
    i, j = torch.triu_indices(N, N)
    A[i, j] = A.T[i, j] = vals # set probabilities in matrix
    A.fill_diagonal_(1) # ensure no vertex is connected to itself
    #A = np.float64(A < prob) # convert probability to edge
    A = np.float64((A < prob) * A) # convert probability to edge
    A = np.ceil(A/prob * scale * step)/step
    return sp.csr_matrix(A)

def connected(gg):
    gga = gg.toarray()
    idx = np.array([0]) #set([0])
    conn = True
    for i in range(gg.shape[0]):
        if i >= idx.shape[0]:
            conn = False
            #print("Not connected")
            break;
        new_idx = np.arange(gg.shape[0])[gga[idx[i]]>0]
        idx, idx_o = np.unique(np.append(idx, new_idx), return_index=True)
        idx = idx[idx_o.argsort()]
    return conn

def cut_partitions(A,Y):
    if not torch.is_tensor(A):
        A = torch.from_numpy(A.toarray()).float().to(device)
    p_tst = test_partition(Y)
    partitions = []
    for i in range(Y.shape[1]):
        partitions.append(A[p_tst==i][:,p_tst==i])
    return partitions

def cuts(A, Y):
    partitions = test_partition(Y)
    cut = torch.zeros(Y.shape[1]).to(device)
    if not torch.is_tensor(A):#isinstance(A, np.ndarray):
        Ac = torch.from_numpy(A.toarray()).to(device)
    else:
        Ac = A.to(device)
    Ac.requires_grad_(True)
    for i in range(Y.shape[1]):
        cut[i] = Ac[partitions==i][:,partitions!=i].sum()#.to(device)
    # force it to always cut, so that it doesnt let partitions empty
    return cut#.maximum(torch.ones(1).to(device)*A.sum(axis=1).min())#*A[A>0].min().to(device))

def volume(A, Y, e_min = 0): 
    # e_min was going to serve as punishment for leaving empty but i changed how it works
    partitions = test_partition(Y)
    vol = torch.zeros(Y.shape[1]).to(device)
    if not torch.is_tensor(A):#isinstance(A, np.ndarray):
        Ac = torch.from_numpy(A.toarray()).to(device)
    else:
        Ac = A.to(device)
    for i in range(Y.shape[1]):
        vol[i] = Ac[partitions==i].sum() # [:,partitions==1]# the second bracket not sure if needed
    # partition must always have something, so that it doesnt let partitions empty
    return vol#.maximum(torch.ones(1).to(device)*A.sum(axis=1).min()) #+ e_min
    
# not to be used for loss
def NCut(A, Y):
    cut = cuts(A, Y)
    vol = volume(A, Y)
    return (cut / vol).sum()

def balance(Y, test=True):
    n = Y.shape[0]
    g = Y.shape[1]
    partitions = test_partition(Y)
    yt = torch.zeros(Y.shape).to(device)
    if test:
        yt[range(n),partitions] = 1
    else:
        yt = Y
    balancedness = ((torch.ones(n).to(device) @ yt) - n/g).square().sum()
    return balancedness

# generates all rotations
def gen_all_rot(A_og, n_rot=None, plot=False):
    if n_rot is None or A_og.shape[0] < n_rot:
        n_rot = A_og.shape[0]
    adj_all = []
    As_all = []
    A_all = []
    for i in np.arange(0,A_og.shape[0], A_og.shape[0]/n_rot).astype(int):
    #range(0,A_og.shape[0], n_rot):
        A = sp.hstack((A_og[:, i:], A_og[:, :i]), format='csr')
        A = sp.vstack((A[i:, :], A[:i, :]), format='csr')
        if plot:
            plt_graph(A)
        A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
        norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
        adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to(device) # SciPy to Torch sparse
        As = sparse_mx_to_torch_sparse_tensor(A).to(device)  # SciPy to sparse Tensor
        A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to(device)   # SciPy to Torch Tensor
        adj_all.append(adj)
        As_all.append(As)
        A_all.append(A)
    return adj_all, As_all, A_all

def gen_x(A, transformer, lazy=False):
    x = []
    if not isinstance(A, list):
        A = [A] # so that it can handle both single and lists
    for A_i in A:
        if not isinstance(A_i, np.ndarray):
            if torch.is_tensor(A_i):
                A_i = A_i.cpu()
            else:
                A_i = A_i.toarray() # sparse to array
        # not sure why but .components_ isnt working inside function
        #if transformer.components_.shape[1] == A_i.shape[1] and lazy:
        #    x.append(transformer.transform(A_i)) # faster computation but worse representation
        #else:
        xx = torch.from_numpy(transformer.fit_transform(A_i))
        x.append(xx.float().to(device))
    return x
    
def save_all(A_arr, transformer, name):
    adj, Ad, As = A_arr
    x = []
    for Ad_i in Ad:
        x.append(torch.from_numpy(transformer.fit_transform(Ad_i.cpu())).to(device))
    torch.save((adj, Ad, As, x), name)
    return adj, Ad, As, x

def Train_rotate(model_r, x, A, optimizer_r, bal=False, rotate=True, max_epochs = 100):
    '''
    Training Specifications
    '''
    if rotate:
        N = A.shape[0]
        adj, A, _ = gen_all_rot(A)
    else:
        N = 1
        A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
        norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
        adj = [sparse_mx_to_torch_sparse_tensor(norm_adj).to(device)] # SciPy to Torch sparse
        A = [sparse_mx_to_torch_sparse_tensor(A).to(device)]  # SciPy to sparse Tensor
        #A_g = As.to_dense().to(device)   # SciPy to Torch Tensor
        #plt_graph(A_g)
        
    min_loss = 100
    for epoch in range(max_epochs):        
        Y = model_r(x, adj[epoch%N])
        loss = CutLoss.apply(Y,A[epoch%N])
        if bal:
            f_loss = loss + l_balance
        else:
            f_loss = loss
        
        if epoch % 20 == 0:
            #plt_graph(A_g[epoch%N])
            print('Epoch {}:   Loss = {}'.format(epoch, f_loss.item()))
        if loss < min_loss:
            min_loss = f_loss.item()
            torch.save(model_r.state_dict(), "./data/trial_weights_r.pt")
        f_loss.backward()
        optimizer_r.step()
        
def Train_random(model_r, x, A, optimizer_r, prob_gen=0.1, prob_A=0.1, initial=0.1, max_epochs = 100):
    '''
    Training Specifications
    '''
    N = A.shape[0]
    adj_r, A_r, _ = gen_all_rot(A)
    
    min_loss = 100
    for epoch in range(max_epochs):
        if torch.rand(1) < prob_A or (epoch < max_epochs*initial): 
            Y = model_r(x, adj_r[epoch%N])
            loss = CutLoss.apply(Y,A_r[epoch%N])
        else:
            A = gen_graph(N, prob_gen)
            A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
            norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
            adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to(device) # SciPy to Torch sparse
            A = sparse_mx_to_torch_sparse_tensor(A).to(device)  # SciPy to sparse Tensor
            Y = model_r(x, adj)
            loss = CutLoss.apply(Y,A)
            
        # loss = custom_loss(Y, A)
        if epoch % 20 == 0:
            #plt_graph(A_g[epoch%N])
            print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model_r.state_dict(), "./data/trial_weights_rand.pt")
        loss.backward()
        optimizer_r.step()


def Test_rotate(model, x, A, *argv, rotate=True, plot=False, s=None):
    '''
    Test Final Results
    '''
    if s is None:
        model.load_state_dict(torch.load("./data/trial_weights.pt"))
    else:
        if s == 'r' :
            model.load_state_dict(torch.load("./data/trial_weights_r.pt"))
        elif s == 'rand' :
            model.load_state_dict(torch.load("./data/trial_weights_rand.pt"))
        else:
            model.load_state_dict(torch.load(s))
            
    if rotate:
        N = A.shape[0]
        adj, A, A_g = gen_all_rot(A)
    else:
        N = 1
        A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
        norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
        adj = [sparse_mx_to_torch_sparse_tensor(norm_adj).to(device)] # SciPy to Torch sparse
        A = [sparse_mx_to_torch_sparse_tensor(A).to(device)]  # SciPy to sparse Tensor
        #A_g = As.to_dense().to(device)   # SciPy to Torch Tensor
    
    Y = []
    for i in range(N):
        Y1 = model(x, adj[i])
        node_idx = test_partition(Y1)
        print(node_idx)
        if rotate:
            if i:
                # compare with the first result
                init_node_idx = torch.roll(init_node_idx, -1)
                if torch.any(init_node_idx.eq(node_idx)) and not torch.equal(init_node_idx, node_idx):
                    print(f"!!! Mismatch: expected = {init_node_idx}")
                    if plot:
                        plt_graph(A_g[i])
            else:
                init_node_idx = node_idx
                    
        if argv != ():
            if argv[0] == 'debug':
                print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y1,A[i]).item()))
        else:
            print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(CutLoss.apply(Y1,A[i]).item()))
        Y.append(Y1)
    return Y

def Train_arr(model, x, adj, As, optimizer, bal=False, max_epochs = 100, file_s=None, overwrite=False, min_loss=100):
    '''
    Training Specifications
    '''
    N = As[0].shape[0]
    L = len(As)
    loss_train = torch.zeros(L)
    if file_s is None:
        file_s = f'./data/weigths_N{N}_L{L}.pt'
    
    if not overwrite:
        try:
          model.load_state_dict(torch.load(file_s))
        except IOError:
          print("Model not saved before, training from scratch")
    
    #min_loss = 100
    print('Min loss to save is {}'.format(min_loss))
    for epoch in range(max_epochs):
        Y = model(x[epoch%L], adj[epoch%L])
        
        loss = CutLoss.apply(Y,As[epoch%L])
        l_balance = BalanceLoss(Y)
        # l_balance -> "balances" by guessing equally for all partition
        if bal:
            f_loss = loss + l_balance
        else:
            f_loss = loss
        
        loss_train[epoch%L] = f_loss
        avg_loss = loss_train.mean() * max(L/(epoch+1),1)
            
        if epoch % 20 == 0:
            print('Epoch {}:   Loss : avg = {} , NCut = {} , balance = {}'.format(epoch, avg_loss, loss.item(), l_balance.item()))
        if avg_loss < min_loss:
            min_loss = loss.item()
            #print("Better model now has loss of {}".format(min_loss))
            torch.save(model.state_dict(), file_s)
        loss.backward()
        optimizer.step()
        

def Test_custom(model, x, adj, A, weights_file, *argv, type=None):
    '''
    Test Final Results
    '''
    model.load_state_dict(torch.load(weights_file))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    print(node_idx)
    if argv != ():
        if argv[0] == 'debug':
            l_ncut = custom_loss(Y,A).item()
    else:
        l_ncut = CutLoss.apply(Y,A).item()
    l_balance = BalanceLoss(Y).item()
    f_loss = l_balance + l_ncut
    
    print('Using partition : Normalized Cut = {0:.3f} , Balance loss = {0:.3f}'.format(l_ncut, l_balance))
    return Y, f_loss

def LeakyPartition(Y):
    partitions = test_partition(Y)
    yt = torch.zeros(Y.shape).to('cuda')
    yt[range(Y.shape[0]),partitions] = 1
    ytt = ((yt+1)/Y.shape[1] * Y)
    return (ytt.T/ytt.sum(1)).T

def Leaky(Y):
    leaky = nn.LeakyReLU()
    return leaky(Y-1/Y.shape[1])

def StepBal(Y):
    partitions = test_partition(Y)
    yt = torch.zeros(Y.shape).to('cuda')
    yt[range(Y.shape[0]),partitions] = 1
    return yt*Y

def BalanceLoss(Y): # just the normal balance makes the model guess all partitions equally
    #Y = LeakyPartition(Y)      # not helpful at all
    #Y = Leaky(Y)
    #Y = StepBal(Y) # a bit better but still leaves partitions empty
    return (Y.sum(dim=0) - Y.shape[0]/Y.shape[1]).square().sum()



# not mine, but here due to hierarchy
def Train(model, x, adj, A, optimizer, bal=False, max_epochs = 100):
    '''
    Training Specifications
    '''

    #max_epochs = 100
    min_loss = 100
    for epoch in (range(max_epochs)):        
        Y = model(x, adj)
        loss = CutLoss.apply(Y,A)
        #loss = NCut(A, Y)
        # loss = custom_loss(Y, A)
        l_balance = BalanceLoss(Y)
        if bal:
            f_loss = loss + l_balance
        else:
            f_loss = loss
        if epoch % 20 == 0:
            print('Epoch {}:   Loss = NCut {} , Bal {}'.format(epoch, loss.item(), l_balance.item()))
        if f_loss < min_loss:
            min_loss = f_loss.item()
            torch.save(model.state_dict(), "./data/trial_weights.pt")
        f_loss.backward()
        optimizer.step()


def Test(model, x, adj, A, *argv, type=None):
    '''
    Test Final Results
    '''
    model.load_state_dict(torch.load("./data/trial_weights.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    print(node_idx)
    if argv != ():
        if argv[0] == 'debug':
            print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(custom_loss(Y,A).item()))
    else:
        print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(CutLoss.apply(Y,A).item()))
    l_balance = BalanceLoss(Y)
    print('Balance Loss = {0:.3f}'.format(l_balance.item()))
    return Y