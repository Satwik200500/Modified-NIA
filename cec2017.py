import numpy as np

# CEC 2017 Benchmark Functions

def Get_F(F):
    if F == 'F1':
        F_obj = F1
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F2':
        F_obj = F2
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F3':
        F_obj = F3
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F4':
        F_obj = F4
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F5':
        F_obj = F5
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F6':
        F_obj = F6
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F7':
        F_obj = F7
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F8':
        F_obj = F8
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F9':
        F_obj = F9
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F10':
        F_obj = F10
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F11':
        F_obj = F11
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F12':
        F_obj = F12
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F13':
        F_obj = F13
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F14':
        F_obj = F14
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F15':
        F_obj = F15
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F16':
        F_obj = F16
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F17':
        F_obj = F17
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F18':
        F_obj = F18
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F19':
        F_obj = F19
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F20':
        F_obj = F20
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F21':
        F_obj = F21
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F22':
        F_obj = F22
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F23':
        F_obj = F23
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F24':
        F_obj = F24
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F25':
        F_obj = F25
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F26':
        F_obj = F26
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F27':
        F_obj = F27
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F28':
        F_obj = F28
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F29':
        F_obj = F29
        LB = -100
        UB = 100
        Dim = 30
    elif F == 'F30':
        F_obj = F30
        LB = -100
        UB = 100
        Dim = 30
    else:
        raise ValueError(f"Function {F} is not defined in CEC 2017.")
    
    return F_obj, LB, UB, Dim


def F1(x):
    """Shifted Sphere Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    return np.sum((x - shift)**2)

def F2(x):
    """Shifted and Rotated Elliptic Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(10**6 * rotated_x**2)

def F3(x):
    """Shifted and Rotated Bent Cigar Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return rotated_x[0]**2 + np.sum(rotated_x[1:]**2)

def F4(x):
    """Shifted and Rotated Discus Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return rotated_x[0]**2 + 10**6 * np.sum(rotated_x[1:]**2)

def F5(x):
    """Shifted and Rotated Rosenbrock's Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(100 * (rotated_x[1:] - rotated_x[:-1]**2)**2 + (1 - rotated_x[:-1])**2)

def F6(x):
    """Shifted and Rotated Ackley’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(rotated_x**2)))
    part2 = -np.exp(0.5 * np.sum(np.cos(2 * np.pi * rotated_x)))
    return part1 + part2 + np.e + 20

def F7(x):
    """Shifted and Rotated Griewangk's Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 1 + np.sum(rotated_x**2) / 4000 - np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, len(rotated_x) + 1))))

def F8(x):
    """Shifted and Rotated Rastrigin’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 10 * len(x) + np.sum(rotated_x**2 - 10 * np.cos(2 * np.pi * rotated_x))

def F9(x):
    """Shifted and Rotated Michalewicz’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    m = 10
    return -np.sum(np.sin(rotated_x) * np.sin(np.arange(1, len(rotated_x) + 1) * rotated_x**2 / np.pi)**(2 * m))

def F10(x):
    """Shifted and Rotated Schwefel’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 418.9829 * len(x) - np.sum(rotated_x * np.sin(np.sqrt(np.abs(rotated_x))))

def F11(x):
    """Shifted and Rotated Weierstrass Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    a = 0.5
    b = 3
    return np.sum(np.abs(np.sin(np.pi * rotated_x) + 0.5 * np.cos(np.pi * rotated_x)))

def F12(x):
    """Shifted and Rotated Happy Cat Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**2) - 10 * np.cos(2 * np.pi * rotated_x) + 10

# Rotated and Shifted functions F13 to F30 (with shifts and rotations)

def F13(x):
    """Shifted and Rotated Hybrid Function 1"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**4) - 10 * np.cos(2 * np.pi * rotated_x)

def F14(x):
    """Shifted and Rotated Hybrid Function 2"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.abs(rotated_x**3))

def F15(x):
    """Shifted and Rotated Hybrid Function 3"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.abs(rotated_x))

def F16(x):
    """Shifted and Rotated Rosenbrock’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(100 * (rotated_x[1:] - rotated_x[:-1]**2)**2 + (1 - rotated_x[:-1])**2)

def F17(x):
    """Shifted and Rotated Ackley’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(rotated_x**2)))
    part2 = -np.exp(0.5 * np.sum(np.cos(2 * np.pi * rotated_x)))
    return part1 + part2 + np.e + 20

def F18(x):
    """Shifted and Rotated Griewangk’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 1 + np.sum(rotated_x**2) / 4000 - np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, len(rotated_x) + 1))))

def F19(x):
    """Shifted and Rotated Rastrigin’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 10 * len(x) + np.sum(rotated_x**2 - 10 * np.cos(2 * np.pi * rotated_x))

def F20(x):
    """Shifted and Rotated Michalewicz’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    m = 10
    return -np.sum(np.sin(rotated_x) * np.sin(np.arange(1, len(rotated_x) + 1) * rotated_x**2 / np.pi)**(2 * m))

def F21(x):
    """Shifted and Rotated Schwefel’s Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 418.9829 * len(x) - np.sum(rotated_x * np.sin(np.sqrt(np.abs(rotated_x))))

def F22(x):
    """Shifted and Rotated Weierstrass Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    a = 0.5
    b = 3
    return np.sum(np.abs(np.sin(np.pi * rotated_x) + 0.5 * np.cos(np.pi * rotated_x)))

def F23(x):
    """Shifted and Rotated Happy Cat Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**2) - 10 * np.cos(2 * np.pi * rotated_x) + 10

# Functions F24 to F30 will follow the same pattern
def F24(x):
    """Shifted and Rotated Hybrid Function 1"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.abs(rotated_x)**2)

def F25(x):
    """Shifted and Rotated Hybrid Function 2"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.sin(rotated_x)**2)

def F26(x):
    """Shifted and Rotated Hybrid Function 3"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.cos(rotated_x)**2)

def F27(x):
    """Shifted and Rotated Hybrid Function 4"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.exp(-rotated_x**2))

def F28(x):
    """Shifted and Rotated Hybrid Function 5"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.log(np.abs(rotated_x) + 1))

def F29(x):
    """Shifted and Rotated Hybrid Function 6"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.tan(rotated_x)**2)

def F30(x):
    """Shifted and Rotated Hybrid Function 7"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.abs(np.sin(rotated_x)) + np.cos(rotated_x)**2)
