import random
from sympy import symbols, S, sin, cos, tan, Add, Mul, Pow, sec, csc, cot

def generate_random_expression(variables, num_terms=3, max_depth=2):
    # Define available functions and operators
    operators = [Add, Mul, Pow]
    functions = [sin, cos, tan, sec, csc, cot]
    
    def create_leaf():
        if random.random() < 0.7:  
            # 70% chance of being a variable
            return random.choice(variables)
        else: 
            # 30% chance of being a constant
            return S(random.randint(1, 10))

    # Recursive function to build the tree
    def create_node(current_depth):
        # Base case: create a leaf
        if current_depth >= max_depth or random.random() < 0.4:
            return create_leaf()
        
        # Recursive case: create a function or operator node
        choice = random.choice(operators + functions)
        
        if choice in functions:
            # Function nodes have a single child
            return choice(create_node(current_depth + 1))
        else:
            # Operator nodes have two children (for simplicity)
            return choice(create_node(current_depth + 1), create_node(current_depth + 1))
    
    # Combine terms with addition to form the final expression
    terms = [create_node(0) for _ in range(num_terms)]

    return Add(*terms)

if __name__ == '__main__':
    # Example usage
    x, y = symbols('x y')
    expr = generate_random_expression([x, y], num_terms=2, max_depth=3)
    print(f"Generated Expression: {expr}")