import numpy as np

# MULTIMODAL NON-SEPARABLE FUNCTIONS
def ackley(x):
    x = np.array(x)
    d = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e

def cross_in_tray(x):
    x1, x2 = x[0], x[1]
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    return -0.0001 * (np.abs(fact1 * fact2) + 1) ** 0.1

def goldstein_price(x):
    x1, x2 = x[0], x[1]
    return (1 + ((x1 + x2 + 1)**2)*(19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) * \
           (30 + ((2*x1 - 3*x2)**2)*(18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))

def mccormick(x):
    x1, x2 = x[0], x[1]
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1

def schaffer_n2(x):
    x1, x2 = x[0], x[1]
    num = (np.sin(x1**2 - x2**2))**2 - 0.5
    denom = (1 + 0.001*(x1**2 + x2**2))**2
    return 0.5 + num / denom

# -- MULTIMODAL SEPARABLE FUNCTIONS --
def alpine1(x):
    x = np.array(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def bohachevsky(x):
    x1, x2 = x[0], x[1]
    return x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7

def bukin4(x):
    x1, x2 = x[0], x[1]
    return 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10)

def csendes(x):
    x = np.array(x)
    return np.sum(x**6 * (2 + np.sin(1 / x)))

def deb1(x):
    x = np.array(x)
    d = len(x)
    return -1/d * np.sum(np.sin(5 * np.pi * x)**6)

def three_hump_camel(x):
    x1, x2 = x[0], x[1]
    return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2

def booth(x):
    x1, x2 = x[0], x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

# -- UNIMODAL NON-SEPARABLE FUNCTIONS --
def beale(x):
    x1, x2 = x[0], x[1]
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

def dixon_price(x):
    x = np.array(x)
    term1 = (x[0] - 1)**2
    term2 = np.sum([(i+1)*(2*x[i+1]**2 - x[i])**2 for i in range(len(x)-1)])
    return term1 + term2

def matyas(x):
    x1, x2 = x[0], x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

def schwefel_12(x):
    x = np.array(x)
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

def schwefel_222(x):
    x = np.array(x)
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def colville(x):
    x1, x2, x3, x4 = x
    return 100*(x2 - x1**2)**2 + (1 - x1)**2 + 90*(x4 - x3**2)**2 + (1 - x3)**2 + \
           10.1*((x2 - 1)**2 + (x4 - 1)**2) + 19.8*(x2 - 1)*(x4 - 1)

def zakharov(x):
    x = np.array(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x)+1) * x)
    return sum1 + sum2**2 + sum2**4

def quartic(x):
    x = np.array(x)
    return np.sum((x**4) + np.random.rand(len(x)))

def easom(x):
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

def chung_reynolds(x):
    x = np.array(x)
    return np.sum(x**2)**2

# -- UNIMODAL SEPARABLE FUNCTIONS --
def powell_sum(x):
    x = np.array(x)
    return np.sum(np.abs(x)**(np.arange(1, len(x)+1) + 1))

def schumer_steiglitz(x):
    x = np.array(x)
    return np.sum(x**4)

def step(x):
    x = np.array(x)
    return np.sum((np.floor(x + 0.5))**2)

def stepint(x):
    x = np.array(x)
    return 25+np.sum(np.floor(x))

def sum_squares(x):
    x = np.array(x)
    return np.sum([(i+1) * x[i]**2 for i in range(len(x))])

def sphere(x):
    x = np.array(x)
    return np.sum(x**2)

# -- MULTI-OBJECTIVE FUNCTIONS --
def modified_inverted_dtlz1(x):
    x = np.array(x)
    g = 100 * (len(x) - 1 + np.sum((x[1:] - 0.5)**2 - np.cos(20 * np.pi * (x[1:] - 0.5))))
    f = 0.5 * (1 - x[0]) * (1 + g)
    return -f  # invertita per minimizzazione

def modified_inverted_dtlz7(x):
    x = np.array(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h = 1 - (f1 / g)**0.5
    f2 = g * h
    return -f2  # invertita per minimizzazione

# -- FUNCTIONS MAPS --
def functions_map():
    return {
        # MULTIMODAL NON-SEPARABLE FUNCTIONS
        "Ackley": ackley,
        "Cross-in-tray": cross_in_tray,
        "Goldstein-Price": goldstein_price,
        "McCormick": mccormick,
        "Schaffer N2": schaffer_n2,
        
        # MULTIMODAL SEPARABLE FUNCTIONS
        "Alpine1": alpine1,
        "Bohachevsky": bohachevsky,
        "Bukin N4": bukin4,
        "Csendes": csendes,
        "Deb1": deb1,
        "Three-hump Camel": three_hump_camel,
        "Booth": booth,
        
        # UNIMODAL NON-SEPARABLE FUNCTIONS
        "Beale": beale,
        "Dixon-Price": dixon_price,
        "Matyas": matyas,
        "Schwefel 1.2": schwefel_12,
        "Schwefel 2.22": schwefel_222,
        "Colville": colville,
        "Zakharov": zakharov,
        "Quartic": quartic,
        "Easom": easom,
        "Chung Reynolds": chung_reynolds,
        
        # UNIMODAL SEPARABLE FUNCTIONS
        "Powell Sum": powell_sum,
        "Schumer Steiglitz": schumer_steiglitz,
        "Step": step,
        "StepInt": stepint,
        "Sum Squares": sum_squares,
        "Sphere": sphere,
        
        # MULTI-OBJECTIVE FUNCTIONS
        "Modified Inverted DTLZ1": modified_inverted_dtlz1,
        "Modified Inverted DTLZ7": modified_inverted_dtlz7
    }

def fixed_dim_functions_map():
    return {
        # MULTIMODAL NON-SEPARABLE FUNCTIONS (2D fisse)
        'Cross-in-tray': 2,      # usa x[0], x[1]
        'Goldstein-Price': 2,    # usa x[0], x[1] 
        'McCormick': 2,          # usa x[0], x[1]
        'Schaffer N2': 2,        # usa x[0], x[1]
        
        # MULTIMODAL SEPARABLE FUNCTIONS (2D fisse)
        'Bohachevsky': 2,        # usa x[0], x[1]
        'Bukin N4': 2,           # usa x[0], x[1]
        'Three-hump Camel': 2,   # usa x[0], x[1]
        'Booth': 2,              # usa x[0], x[1]
        
        # UNIMODAL NON-SEPARABLE FUNCTIONS
        'Beale': 2,              # usa x[0], x[1]
        'Matyas': 2,             # usa x[0], x[1]
        'Colville': 4,           # usa x1, x2, x3, x4 (4D fissa)
        'Easom': 2,              # usa x[0], x[1]
    }


def best_values_map():
    return {
        # MULTIMODAL NON-SEPARABLE FUNCTIONS
        "Ackley": 0,
        "Cross-in-tray": -2.06261,  # valore minimo circa -2.06261 a (±1.35, ±1.35)
        "Goldstein-Price": 3,       # valore minimo 3 a (0, -1)
        "McCormick": -1.9133,       # valore minimo circa -1.9133 a (-0.54, -1.54)
        "Schaffer N2": 0,           # valore minimo 0 a (0, 0)
        
        # MULTIMODAL SEPARABLE FUNCTIONS
        "Alpine1": 0,
        "Bohachevsky": 0,
        "Bukin N4": 0,
        "Csendes": 0,
        "Deb1": -1,                 # funzione di minimizzazione negativa, minimo -1
        "Three-hump Camel": 0,
        "Booth": 0,
        
        # UNIMODAL NON-SEPARABLE FUNCTIONS
        "Beale": 0,
        "Dixon-Price": 0,
        "Matyas": 0,
        "Schwefel 1.2": 0,
        "Schwefel 2.22": 0,
        "Colville": 0,
        "Zakharov": 0,
        "Quartic": 0,               # con componente stocastica, teorico minimo 0
        "Easom": -1,                # minimo -1 a (π, π)
        "Chung Reynolds": 0,
        
        # UNIMODAL SEPARABLE FUNCTIONS
        "Powell Sum": 0,
        "Schumer Steiglitz": 0,
        "Step": 0,
        "StepInt": None,  # non ha un minimo globale definito
        "Sum Squares": 0,
        "Sphere": 0,
        
        # MULTI-OBJECTIVE FUNCTIONS
        "Modified Inverted DTLZ1": 0,  # valori invertiti per minimizzazione
        "Modified Inverted DTLZ7": 0   # valori invertiti per minimizzazione
    }

def bounds_map():
    return {
        # MULTIMODAL NON-SEPARABLE FUNCTIONS
        "Ackley": (-5, 5),
        "Cross-in-tray": (-10, 10),
        "Goldstein-Price": (-2, 2),
        "McCormick": ([-1.5, -3], [4, 4]),
        "Schaffer N2": (-100, 100),
        
        # MULTIMODAL SEPARABLE FUNCTIONS
        "Alpine1": (-10, 10),
        "Bohachevsky": (-100, 100),
        "Bukin N4": (-15, 3),
        "Csendes": (-1, 1),
        "Deb1": (-1, 1),
        "Three-hump Camel": (-5, 5),
        "Booth": (-10, 10),
        
        # UNIMODAL NON-SEPARABLE FUNCTIONS
        "Beale": (-10, 10),
        "Dixon-Price": (-10, 10),
        "Matyas": (-10, 10),
        "Schwefel 1.2": (-65.539, 65.539),
        "Schwefel 2.22": (-10, 10),
        "Colville": (-10, 10),
        "Zakharov": (-5, 10),
        "Quartic": (-1.28, 1.28),
        "Easom": (-100, 100),
        "Chung Reynolds": (-100, 100),
        
        # UNIMODAL SEPARABLE FUNCTIONS
        "Powell Sum": (-1, 1),
        "Schumer Steiglitz": (-100, 100),
        "Step": (-100, 100),
        "StepInt": (-5.12, 5.12),
        "Sum Squares": (-10, 10),
        "Sphere": (-5.12, 5.12),
        
        # MULTI-OBJECTIVE FUNCTIONS
        "Modified Inverted DTLZ1": (0, 1),
        "Modified Inverted DTLZ7": (0, 1)
    }
