import random
from sympy import symbols, S, sin, cos, tan, Add, Mul, Pow, sec, csc, cot, latex

def generate_random_expression(variables, num_terms=3, max_depth=2):

    # Ensure all variables are SymPy symbols
    variables = [symbols(v) if isinstance(v, str) else v for v in variables]

    operators = ["add", "mul", "pow"]
    functions = [sin, cos, tan, sec, csc, cot]

    def create_leaf():
        if random.random() < 0.7:
            return random.choice(variables)  # variable
        else:
            return S(random.randint(1, 10))  # constant

    # 🌟 MODIFICATION 1: Safe exponent only returns constants (integers 1 to 5).
    # This prevents 'x^y' or 'x^(sin(x))' cases.
    def safe_exponent():
        return S(random.randint(1, 5))

    # Helper to check if an expression is a variable (SymPy symbol)
    def is_variable_expr(expr):
        return any(v in expr.free_symbols for v in variables)

    def create_node(current_depth):
        if current_depth >= max_depth or random.random() < 0.4:
            return create_leaf()

        choice = random.choice(operators + ["func"])

        # function node
        if choice == "func":
            func = random.choice(functions)
            return func(create_node(current_depth + 1))

        # operator node
        left = create_node(current_depth + 1)
        right = create_node(current_depth + 1)

        if choice == "add":
            return left + right

        elif choice == "mul":
            return left * right

        elif choice == "pow":
            base = left
            
            if is_variable_expr(base):
                exponent = safe_exponent() 
            else:
                exponent = safe_exponent()

            return Pow(base, exponent) 

    terms = [create_node(0) for _ in range(num_terms)]
    expr = Add(*terms)
    
    # Return the SymPy expression, its string representation, and its LaTeX representation
    return expr, str(expr), latex(expr)


if __name__ == '__main__':
    x, y = symbols('x y')
    # Update unpacking here too for testing
    expr, expr_str, expr_latex = generate_random_expression([x, y], num_terms=2, max_depth=3)
    print(f"Generated Expression: {expr}")
    print(f"Generated Expression String: {expr_str}")
    print(f"Generated Expression LaTeX: {expr_latex}")