import numpy as np
import matplotlib.pyplot as plt

# Function to get the function and bounds (this is a placeholder, replace with your actual function)
def Get_F(func_name):
    if func_name == 'F1':
        return (-100, 100, 2, lambda x: np.sum(x**2))  # Example for F1: Sphere function
    elif func_name == 'F2':
        return (-10, 10, 2, lambda x: np.sum(x**2))
    # Add cases for other functions as needed
    # Return appropriate lower bound, upper bound, dimension, and function handle
    return (-5, 5, 2, lambda x: np.sum(x**2))  # Default case

def func_plot(func_name):
    # Get the lower bound, upper bound, dimensions, and objective function
    LB, UB, Dim, F_obj = Get_F(func_name)
    
    # Generate x and y ranges based on the function name
    if func_name == 'F1':
        x = np.arange(-100, 100, 2)
        y = x
    elif func_name == 'F2':
        x = np.arange(-100, 100, 2)
        y = x
    elif func_name == 'F3':
        x = np.arange(-100, 100, 2)
        y = x
    elif func_name == 'F4':
        x = np.arange(-100, 100, 2)
        y = x
    elif func_name == 'F5':
        x = np.arange(-200, 200, 2)
        y = x
    elif func_name == 'F6':
        x = np.arange(-100, 100, 2)
        y = x
    elif func_name == 'F7':
        x = np.arange(-1, 1, 0.03)
        y = x
    elif func_name == 'F8':
        x = np.arange(-500, 500, 10)
        y = x
    elif func_name == 'F9':
        x = np.arange(-5, 5, 0.1)
        y = x
    elif func_name == 'F10':
        x = np.arange(-20, 20, 0.5)
        y = x
    elif func_name == 'F11':
        x = np.arange(-500, 500, 10)
        y = x
    elif func_name == 'F12':
        x = np.arange(-10, 10, 0.1)
        y = x
    elif func_name == 'F13':
        x = np.arange(-5, 5, 0.08)
        y = x
    elif func_name == 'F14':
        x = np.arange(-100, 100, 2)
        y = x
    elif func_name == 'F15':
        x = np.arange(-5, 5, 0.1)
        y = x
    elif func_name == 'F16':
        x = np.arange(-1, 1, 0.01)
        y = x
    elif func_name == 'F17':
        x = np.arange(-5, 5, 0.1)
        y = x
    elif func_name == 'F18':
        x = np.arange(-5, 5, 0.06)
        y = x
    elif func_name == 'F19':
        x = np.arange(-5, 5, 0.1)
        y = x
    elif func_name == 'F20':
        x = np.arange(-5, 5, 0.1)
        y = x
    elif func_name == 'F21':
        x = np.arange(-5, 5, 0.1)
        y = x
    elif func_name == 'F22':
        x = np.arange(-5, 5, 0.1)
        y = x
    elif func_name == 'F23':
        x = np.arange(-5, 5, 0.1)
        y = x
    
    # Create a grid mesh from x and y
    X, Y = np.meshgrid(x, y)
    L = len(x)
    f = np.zeros((L, L))
    
    # Calculate the function values
    for i in range(L):
        for j in range(L):
            if func_name not in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23']:
                f[i, j] = F_obj(np.array([x[i], y[j]]))
            elif func_name == 'F15':
                f[i, j] = F_obj(np.array([x[i], y[j], 0, 0]))
            elif func_name == 'F19':
                f[i, j] = F_obj(np.array([x[i], y[j], 0]))
            elif func_name == 'F20':
                f[i, j] = F_obj(np.array([x[i], y[j], 0, 0, 0, 0]))
            elif func_name in ['F21', 'F22', 'F23']:
                f[i, j] = F_obj(np.array([x[i], y[j], 0, 0]))

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, f, cmap='viridis')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Objective function value')
    ax.set_title(f"Surface plot for {func_name}")
    plt.show()

# Example usage
func_plot('F1')
