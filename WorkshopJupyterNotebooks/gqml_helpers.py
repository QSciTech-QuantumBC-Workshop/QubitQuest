import numpy as np
import qiskit as qk

def who_won(position):
    eight_configurations = [[0,1,2],[0,3,6],[0,4,8],[3,4,5],[1,4,7],[6,4,2],[2,5,8],[6,7,8]]
    for conf in eight_configurations:
        total = 0
        for i in range(3):
            total += position[conf[i]]
        if total == 3:
            return [1,-1,-1]
        elif total == -3:
            return [-1,-1,1]
    return [-1,1,-1]

def split_data(train_ratio, data, labels):
    n_train = int(train_ratio*len(labels))
    train_inds = np.random.choice(len(labels), n_train, replace = False)
    train_pos = [data[i] for i in train_inds]
    train_classes = [labels[i] for i in train_inds]
    a = {i for i in range(len(labels))}
    for ind in train_inds:
        a.remove(ind)
    test_inds = list(a)
    test_pos = [data[i] for i in test_inds]
    test_classes = [labels[i] for i in test_inds]
    return train_pos, train_classes, test_pos, test_classes

#single qubit gates
def make_c(theta_1, theta_2):
    c = qk.QuantumCircuit(9)
    for qb in [0,2,6,8]:
        c.rx(theta_1, qb)
        c.ry(theta_2, qb)
    return c

def make_e(theta_1, theta_2):
    e = qk.QuantumCircuit(9)
    for qb in [1,3,5,7]:
        e.rx(theta_1, qb)
        e.ry(theta_2, qb)
    return e

def make_m(theta_1, theta_2):
    m = qk.QuantumCircuit(9)
    m.rx(theta_1,4)
    m.ry(theta_2,4)
    return m

#Entangling gates

def make_o(theta):
    o = qk.QuantumCircuit(9)
    for qb_pair in [[0,1], [0,3], [2,1], [2,5], [6,3], [6,7], [8,5], [8,7]]:
        o.cry(theta,qb_pair[0], qb_pair[1])
    return o
def make_inn(theta):
    inn = qk.QuantumCircuit(9)
    for control_qb in [1,3,5,7]:
        inn.cry(theta,control_qb,4)
    return inn
def make_d(theta):
    d = qk.QuantumCircuit(9)
    for target_qb in [0,2,6,8]:
        d.cry(theta,4,target_qb)
    return d

#circuit

def make_full_circuit(l,p, data_param_dict):
    thetas = [qk.circuit.Parameter(f'theta_{i}') for i in range(9*p*l)]
    qc = qk.QuantumCircuit(9)
    curr_theta = 0
    for i in range(l):
        for j in range(9):
            qc.rx(2 * np.pi / 3 * data_param_dict[f'x_{j}'], j)
        for j in range(p):
            qc.append(make_c(thetas[curr_theta], thetas[curr_theta+1]), range(9))
            qc.append(make_e(thetas[curr_theta+2], thetas[curr_theta+3]), range(9))
            qc.append(make_m(thetas[curr_theta+4],thetas[curr_theta+5]), range(9))
            qc.append(make_d(thetas[curr_theta + 6]), range(9))
            qc.append(make_o(thetas[curr_theta+7]), range(9))
            qc.append(make_inn(thetas[curr_theta + 8]), range(9))
            curr_theta += 9
    return qc

#measurement functions

def measure_z_mid(circuit):
    obs = qk.quantum_info.SparsePauliOp('IIIIZIIII')
    estimator = qk.primitives.Estimator()
    job = estimator.run(circuit, obs)
    result = job.result()
    return result.values[0]
def measure_z_corners(circuit):
    estimator = qk.primitives.Estimator()
    observables = [qk.quantum_info.SparsePauliOp('ZIIIIIIII'), qk.quantum_info.SparsePauliOp('IIZIIIIII'), qk.quantum_info.SparsePauliOp('IIIIIIZII'), qk.quantum_info.SparsePauliOp('IIIIIIIIZ')]
    total = 0
    for obs in observables:
        job = estimator.run(circuit, obs)
        total += job.result().values[0]
    return total/4
def measure_z_edges(circuit):
    estimator = qk.primitives.Estimator()
    observables = [qk.quantum_info.SparsePauliOp('IZIIIIIII'), qk.quantum_info.SparsePauliOp('IIIZIIIII'), qk.quantum_info.SparsePauliOp('IIIIIZIII'), qk.quantum_info.SparsePauliOp('IIIIIIIZI')]
    total = 0
    for obs in observables:
        job = estimator.run(circuit, obs)
        total += job.result().values[0]
    return total/4


def score(qc, params, test_pos, test_labels):
    success = 0
    for i, pos in enumerate(test_pos):
        to_bind = np.concatenate((params,pos))
        target = test_labels[i]
        qc_temp = qc.bind_parameters(to_bind)
        label = np.array([measure_z_corners(qc_temp), measure_z_mid(qc_temp), measure_z_edges(qc_temp)])
        if np.argmax(target) == np.argmax(label):
            success += 1
    return success/(len(test_pos))