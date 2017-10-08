import cvxpy as cvx
import numpy as np

# O(nC*nlg(n))
def fastAssign(Y, C, maxC=500, minC=1):
    from collections import Counter
    from operator import itemgetter

    n, nF = Y.shape
    nC = C.shape[0]
    assert C.shape[1] == nF

    YCdist = list()
    for i in range(n):
        for j in range(nC):
            YCdist.append((i, j, np.linalg.norm(Y[i] - C[j], ord=2)))
    YCdist.sort(key=itemgetter(2))

    cluster = Counter(range(nC))
    for j in range(nC):
        cluster[j] = 0

    assign = -1 * np.ones(n, dtype=int)

    # redo the preassign according to minC
    for tup in YCdist:
        i, j, dist = tup

        if assign[i] < 0 and cluster[j] < minC:
            assign[i] = j
            cluster[j] += 1

    # fill the remained the slots
    for tup in YCdist:
        i, j, dist = tup

        if assign[i] < 0 and cluster[j] < maxC:
            assign[i] = j
            cluster[j] += 1

    assert np.min(assign) >= 0

    return assign

def eqAssign(Y, C, minC=1, accurateAssign=False):

    n, nF = Y.shape
    nC = C.shape[0]
    assert C.shape[1] == nF

    YCdist = np.zeros((n, nC))
    for i in range(n):
        for j in range(nC):
            YCdist[i, j] = np.linalg.norm(Y[i] - C[j], ord=2)

    if accurateAssign:
        X = cvx.Int(n, nC)
    else:
        X = cvx.Variable(n, nC)

    obj = cvx.sum_entries(cvx.mul_elemwise(YCdist, X))

    cons = [cvx.sum_entries(X, axis=1) == 1,
            cvx.sum_entries(X, axis=0) >= minC,
            0 <= X, X <= 1]

    prob = cvx.Problem(cvx.Maximize(obj), cons)

    # Solve
    if accurateAssign:
        prob.solve(solver=cvx.ECOS_BB) #, mi_max_iters=100
        Xopt = X.value
        assign = np.zeros(n)
        for i in range(n):
            for j in range(nC):
                if Xopt[i, j] > 0.9:
                    assign[i] = j
                    break
    else:
        prob.solve(solver=cvx.ECOS)  # , mi_max_iters=100
        Xopt = X.value
        #assign = np.argmax(Xopt, axis=1).ravel()
        assign = np.zeros(n, dtype=int)
        for i in range(n):
            maxX = 0
            maxJ = 0
            for j in range(nC):
                if Xopt[i, j] > maxX:
                    maxX = Xopt[i, j]
                    maxJ = j
            assign[i] = np.round(maxJ)

    return assign, Xopt, prob.status



def WMD(wvs1, wvs2, d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    assert wvs1.shape[0] == n1
    assert wvs2.shape[0] == n2
    assert wvs1.shape[1] == wvs2.shape[2]

    wwdist = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            wwdist[i, j] = np.linalg.norm(wvs1[i] - wvs2[j], ord=2)

    T = cvx.Variable(n1, n2)

    obj = cvx.sum_entries(cvx.mul_elemwise(wwdist, T))

    cons = [cvx.sum_entries(T, axis=1) == d1,
            cvx.sum_entries(T, axis=0) >= d2,
            0 <= T]

    prob = cvx.Problem(cvx.Maximize(obj), cons)

    # Solve
    prob.solve(solver=cvx.ECOS)  # , mi_max_iters=100
    Topt = T.value
    senDiff = prob.value

    return senDiff, Topt, prob.status
