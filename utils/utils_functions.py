import numpy as np

def control_property(A, B):
    controllability_matrix = B
    controllability_column = B
    for _ in range(A.shape[0]-1):
        controllability_column = A @ controllability_column
        controllability_matrix = np.c_[controllability_matrix,controllability_column]
    if np.linalg.matrix_rank(controllability_matrix) == A.shape[0]:
        print("The system is controllable.")
    else:
        print("The system is uncontrollable, the rank of the controllability matrix is "+str(np.linalg.matrix_rank(controllability_matrix))+".")

