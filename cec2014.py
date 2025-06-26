import numpy as np

# CEC 2014 Benchmark Functions

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
        LB = -32
        UB = 32
        Dim = 30
    elif F == 'F6':
        F_obj = F6
        LB = -0.5
        UB = 0.5
        Dim = 30
    elif F == 'F7':
        F_obj = F7
        LB = -600
        UB = 600
        Dim = 30
    elif F == 'F8':
        F_obj = F8
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F9':
        F_obj = F9
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F10':
        F_obj = F10
        LB = -500
        UB = 500
        Dim = 30
    elif F == 'F11':
        F_obj = F11
        LB = -500
        UB = 500
        Dim = 30
    elif F == 'F12':
        F_obj = F12
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F13':
        F_obj = F13
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F14':
        F_obj = F14
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F15':
        F_obj = F15
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F16':
        F_obj = F16
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F17':
        F_obj = F17
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F18':
        F_obj = F18
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F19':
        F_obj = F19
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F20':
        F_obj = F20
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F21':
        F_obj = F21
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F22':
        F_obj = F22
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F23':
        F_obj = F23
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F24':
        F_obj = F24
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F25':
        F_obj = F25
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F26':
        F_obj = F26
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F27':
        F_obj = F27
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F28':
        F_obj = F28
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F29':
        F_obj = F29
        LB = -5
        UB = 5
        Dim = 30
    elif F == 'F30':
        F_obj = F30
        LB = -5
        UB = 5
        Dim = 30
    else:
        raise ValueError(f"Function {F} is not defined in CEC 2014.")
    
    return F_obj, LB, UB, Dim


def F1(x):
    """Sphere Function"""
    return np.sum(x**2)

def F2(x):
    """Rosenbrock’s Function"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def F3(x):
    """Rotated Hyper-Ellipsoid Function"""
    return np.sum(10**6 * x**2)

def F4(x):
    """Step Function"""
    return np.sum(np.floor(x))

def F5(x):
    """Griewangk’s Function"""
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def F6(x):
    """Rastrigin’s Function"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def F7(x):
    """Ackley’s Function"""
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2)))
    part2 = -np.exp(0.5 * np.sum(np.cos(2 * np.pi * x)))
    return part1 + part2 + np.e + 20

def F8(x):
    """Schwefel’s Function"""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    """Michalewicz’s Function"""
    m = 10
    return -np.sum(np.sin(x) * np.sin(np.arange(1, len(x) + 1) * x**2 / np.pi)**(2 * m))

def F10(x):
    """Generalized Schwefel’s Function"""
    return np.sum(np.abs(x))

def F11(x):
    """Modified Ackley’s Function"""
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20

def F12(x):
    """Balanced Function"""
    return np.sum(np.sin(x**2)**2 - 0.5)**2

def F13(x):
    """High Conditioned Elliptic Function"""
    return np.sum(10**6 * x**2)

def F14(x):
    """Disk-Shaped Function"""
    return np.sqrt(np.sum(x**2))

def F15(x):
    """Rosenbrock’s Function with Rotation"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def F16(x):
    """Discontinuous Function"""
    return np.sum(np.floor(x)) * np.sin(np.sum(x))

def F17(x):
    """Exponential Function"""
    return np.sum(np.exp(x)) + np.cos(x)

def F18(x):
    """Piecewise Linear Function"""
    return np.sum(np.abs(x))

def F19(x):
    """Shifted Sphere Function"""
    shift = np.random.uniform(-5, 5, len(x))  # Shift values
    return np.sum((x - shift)**2)

def F20(x):
    """Cigar Function"""
    return np.sum(x**2) + np.sum(10**6 * x**2)

def F21(x):
    """Rotation Function"""
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    return np.sum(np.dot(x, A)**2)

def F22(x):
    """Asymetric Function"""
    return np.sum(x**3 + np.sin(x)**2)

def F23(x):
    """Rotated Ackley Function"""
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x, A)  # Apply rotation
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(rotated_x**2)))
    part2 = -np.exp(0.5 * np.sum(np.cos(2 * np.pi * rotated_x)))
    return part1 + part2 + np.e + 20

def F24(x):
    """Shifted Rastrigin Function"""
    shift = np.random.uniform(-5, 5, len(x))  # Shift values
    return 10 * len(x) + np.sum((x - shift)**2 - 10 * np.cos(2 * np.pi * (x - shift)))

def F25(x):
    """Shifted Schwefel Function"""
    shift = np.random.uniform(-5, 5, len(x))  # Shift values
    return 418.9829 * len(x) - np.sum((x - shift) * np.sin(np.sqrt(np.abs(x - shift))))

def F26(x):
    """Shifted and Rotated Griewangk Function"""
    shift = np.random.uniform(-5, 5, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 1 + np.sum(rotated_x**2) / 4000 - np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, len(rotated_x) + 1))))

def F27(x):
    """Shifted and Rotated Rastrigin Function"""
    shift = np.random.uniform(-5, 5, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 10 * len(x) + np.sum(rotated_x**2 - 10 * np.cos(2 * np.pi * rotated_x))

def F28(x):
    """Shifted Ackley Function"""
    shift = np.random.uniform(-5, 5, len(x))  # Shift values
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum((x - shift)**2))) - np.exp(0.5 * np.sum(np.cos(2 * np.pi * (x - shift)))) + np.e + 20

def F29(x):
    """Rotated Elliptic Function"""
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    return np.sum(np.dot(x, A)**2)

def F30(x):
    """Shifted Rotated Griewangk Function"""
    shift = np.random.uniform(-5, 5, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 1 + np.sum(rotated_x**2) / 4000 - np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, len(rotated_x) + 1))))

