import numpy as np


# --- T-NORMS (Triangular Norms) ---


def t_M(x, y):
    """T_M: Minimum (eq. 2)"""
    return np.minimum(x, y)


def t_P(x, y):
    """T_P: Product (eq. 3)"""
    return x * y


def t_L(x, y):
    """T_L: Lukasiewicz t-norm (eq. 4)"""
    return np.maximum(0, x + y - 1)


def t_D(x, y):
    """T_D: Drastic product (eq. 5)"""

    return np.where(x == 1, y, np.where(y == 1, x, 0))


def t_FD(x, y):
    """T_FD: Fodor t-norm (eq. 6)"""

    return np.where(x + y <= 1, 0, np.minimum(x, y))


# --- S-CONORMS (Triangular Conorms) ---


def s_M(x, y):
    """S_M: Maximum (eq. 7)"""
    return np.maximum(x, y)


def s_P(x, y):
    """S_P: Probabilistic sum (eq. 8)"""
    return x + y - (x * y)


def s_L(x, y):
    """S_L: Lukasiewicz t-conorm (eq. 9)"""
    return np.minimum(1, x + y)


def s_D(x, y):
    """S_D: Drastic sum (eq. 10)"""

    return np.where(x == 0, y, np.where(y == 0, x, 1))


def s_FD(x, y):
    """S_FD: Fodor t-conorm (eq. 11)"""
    # 1 if x+y >= 1, else max(x,y)
    return np.where(x + y >= 1, 1, np.maximum(x, y))

# --- INDUCED IMPLICATIONS ---

def imp_T_M(a, b):
    """Implication induced by T_M (Godel implication)"""

    return np.where(a <= b, 1.0, b)


def imp_T_P(a, b):
    """Implication induced by T_P (Goguen implication)"""

    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(a <= b, 1.0, b / a)
        res = np.where(a == 0, 1.0, res)
    return res


def imp_T_L(a, b):
    """Implication induced by T_L (Lukasiewicz implication)"""

    return np.minimum(1, 1 - a + b)


def imp_T_FD(a, b):
    """Implication induced by T_FP (Fodor implication)"""

    return np.where(a <= b, 1.0, np.maximum(1 - a, b))

# --- Dual IMPLICATIONS ---

def dual_imp_S_M(a, b):
    """Dual implication induced by T_M (Minimum)"""
    return np.where(a >= b, 0.0, b)

def dual_imp_S_P(a, b):
    """Dual implication induced by T_P (Product)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(a >= b, 0.0, (b - a) / (1.0 - a))
    return res

def dual_imp_S_L(a, b):
    """Dual implication induced by T_L (Lukasiewicz)"""
    return np.maximum(0.0, b - a)

# --- CUSTOM OPERATIONS FOR PAPER EXAMPLE 11 ---
def op_EX1(a, b):
    """Operation: a * b = a * min(a, b)"""
    return a * np.minimum(a, b)

def IMP_EX1(a, b):
    """Implication for Example 11"""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(a**2 <= b, 1.0, b / a)
    return res

def DIMP_EX1(a, b):
    """Dual implication for Example 11"""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(a**2 > b, b / a, a)
    return res

NORM_MAP = {
    # T-Norms
    'T_M': t_M,  # Minimum
    'T_P': t_P,  # Product
    'T_L': t_L,  # Lukasiewicz
    'T_D': t_D,  # Drastic
    'T_FD': t_FD,  # Fodor

    # S-Conorms
    'S_M': s_M,  # Maximum
    'S_P': s_P,  # Probabilistic
    'S_L': s_L,  # Lukasiewicz
    'S_D': s_D,  # Drastic
    'S_FD': s_FD,  # Fodor

    # Implications
    'IMP_T_M': imp_T_M,  # Induced by T_M
    'IMP_T_P': imp_T_P,  # Induced by T_P
    'IMP_T_L': imp_T_L,  # Induced by T_L
    'IMP_T_FD': imp_T_FD,  # Induced by T_FP (Fodor)

    # Dual Implications
    'DIMP_S_M': dual_imp_S_M,
    'DIMP_S_P': dual_imp_S_P,
    'DIMP_S_L': dual_imp_S_L,

    # Custom operations
    'OP_EX1': op_EX1,
    'IMP_EX1': IMP_EX1,
    'DIMP_EX1': DIMP_EX1,
}


def get_operator(identifier):
    """Retrieves a function by the paper's notation string."""
    if callable(identifier):
        return identifier
    if isinstance(identifier, str):

        key = identifier.upper()

        if key == 'MIN': key = 'T_M'
        if key == 'MAX': key = 'S_M'

        if key in NORM_MAP:
            return NORM_MAP[key]

    raise ValueError(f"Norm '{identifier}' not found in registry. Please use paper notation (e.g., 'T_M', 'S_L').")