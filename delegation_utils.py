import numpy as np
import itertools
import more_itertools
from tqdm import tqdm

def retained(Y, h):
    return [cat for cat in itertools.product(range(2),repeat=h) if cat not in Y]

# construct full vector v where v[S] = x, v[S^C] = z
def state(x,z,S,d):
    v = np.zeros(d, np.int64)
    v[list(S)] = x
    S_comp = np.setdiff1d(range(d), S)
    v[S_comp] = z
    return v

def yielded(v, Y, h):
    vh = v[np.arange(h)]
    for arr in Y:
        arr = np.array(arr, np.int64)
        if not np.any(vh - arr):
            return True
    return False

def get_human_errors(f, h, d):
    other_feat = tuple(np.arange(h,d))
    errors = (f - np.mean(f, axis=other_feat, keepdims=1))**2
    return np.sum(errors, axis=other_feat)

def categories_actually_not_delegated_in(h, d, f, h_err, model, I_M):
    adopt = np.zeros(2*np.ones(h, np.int64))
    errors = h_err.copy()
    dropped_cats = []
    for cat in itertools.product(range(2),repeat=h):
        cat_h_err = h_err[cat]
        cat_m_err = 0
        for z in itertools.product(range(2), repeat=d-h):
            v = state(cat, z, np.arange(h), d)
            f_v = f[tuple(v)]
            x = v[list(I_M)]
            m_v = model[tuple(x)]
            cat_m_err += (f_v - m_v)**2
        
        if cat_m_err < cat_h_err:
            adopt[cat] = 1
            errors[cat] = cat_m_err
        else:
            dropped_cats.append(cat)
    return adopt, errors, dropped_cats

def get_model_and_error(m,h,d,f,S,Y,h_errors):
    m_err = 0
    fM = np.zeros(2*np.ones(m, np.int64))
    
    for x in itertools.product(range(2), repeat=m): # for each algorithm category

        values = []
        for z in itertools.product(range(2), repeat=d-m): # for each state in the category
            v = state(x,z,S,d) # get the full state vector
            if not yielded(v, Y, h): # ignore yielded states
                values.append(f[tuple(v)])

        # compute the best prediction on the non-ignored states for this category
        count = len(values)
        avg = np.mean(values) if count > 0 else 0 

        # update model
        fM[tuple(x)] = avg

        # compute machine errors
        if count > 0:
            diffs = np.array([val - avg for val in values])
            m_err += np.sum(diffs**2)
        
    # add human errors for the dropped categories
    h_err = 0
    for y in Y:
        h_err += h_errors[tuple(y)]

    return h_err + m_err, fM

# the best algorithm that drops categories
def find_best_model(m,h,s,d,f):
    human_errors = get_human_errors(f,h,d)

    m0 = h-s
    I_M = np.arange(m0, m0+m)
    
    best_err = np.Inf
    best_model = None
    best_Y = None
    for Y in more_itertools.powerset(itertools.product(range(2),repeat=h)): # Y is a set of human categories the machine does not retain
        err, model = get_model_and_error(m,h,d,f,I_M,Y, human_errors)

        # update best so far
        if err < best_err:
            best_err = err
            best_model = model
            best_Y = Y

    return best_model, best_err, best_Y, human_errors

def find_delegate_iteratively(h,m,s,d,f, quiet=True):
    human_errors = get_human_errors(f,h,d)
    
    m0 = h-s
    I_M = np.arange(m0, m0+m)
    
    # start oblivious machine
    Y = []
    improving = True
    best_loss = np.Inf
    best_model = None
    count = 0
    while improving:
        if not quiet:
            print("START OF ITERATION")
            print("Current R:", retained(Y, h))
        _, model = get_model_and_error(m,h,d,f,I_M,Y,human_errors)
        adopt, errors, Y = categories_actually_not_delegated_in(h,d,f, human_errors, model, I_M)
        err = np.sum(errors)
        if not quiet:
            print("Model designed for R:", model)
            print("Categories model is adopted:", retained(Y, h))
            print(errors)
            print("Team performance:", err)
            print()
        if err < best_loss:
            best_model = model
            best_loss = err
            count += 1
        else:
            improving = False

    return best_model, best_loss, Y, human_errors, count

    def find_Rk(u, k):
    min_var = np.Inf
    best_t = None
    for t in range(len(u)-k+1):
        u_R = u[t:t+k]
        variance = np.var(u_R)
        if variance < min_var:
            min_var = variance
            best_t = t
    return best_t, min_var

def find_best_model_separable_alg(u_unsrt,w,h):
    u_order = np.argsort(u_unsrt)
    u = np.array(u_unsrt)[u_order]
    
    H = len(u)

    var_w = np.var(w)

    best_loss = np.Inf
    best_k = None
    best_tk = None
    for k in range(H+1):
        if k == 0:
            loss = var_w
            best_t = None
        else:
            best_t, min_var_uk = find_Rk(u, k)
            loss = (1-(k/H))*var_w + (k/H)*min_var_uk

        if loss < best_loss:
            best_loss = loss
            best_tk = best_t
            best_k = k

    if best_k == 0:
        R = []
    else:
        R = np.sort(np.arange(H)[u_order][best_tk:best_tk+best_k])

    Y = [Y for i, Y in enumerate(itertools.product(range(2),repeat=h)) if i not in R]

    return Y