import re
import time
import tracemalloc
import logging
from typing import List, Dict, Optional, Tuple

# --- Logger Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# A set of known mathematical functions (used as node values for unary operations).
SUPPORTED_FUNCTIONS = {"sin", "cos", "tan", "sec", "csc", "cot", "exp", "sqrt"}

# --- Directed Acyclic Graph (DAG) Node ---
class DAGNode:
    def __init__(self, value, children: Optional[Tuple['DAGNode', ...]] = None):
        # Value is the operator ('+', '*', 'sin', etc.) or the literal value (1.0, 'x', etc.)
        self.value = value 
        # Children must be a tuple for hashability and immutability
        self.children = children or tuple()

    def __hash__(self):
        # Hash based on value and children tuple (which is hashable)
        return hash((self.value, self.children))

    def __eq__(self, other):
        if not isinstance(other, DAGNode):
            return NotImplemented
        return self.value == other.value and self.children == other.children

    def __repr__(self):
        if not self.children:
            return str(self.value)
        # Use abbreviated repr for children to prevent recursion depth issues in logging
        child_reprs = ', '.join(f"<{c.value}>" for c in self.children)
        return f"DAGNode({self.value}, [{child_reprs}])"

# --- Tokenizer and Parser (Adapted for DAG Canonicalization) ---
TOKEN_NUMBER = 'NUMBER'
TOKEN_SYMBOL = 'SYMBOL'
TOKEN_FUNCTION = 'FUNCTION'
TOKEN_OPERATOR = 'OPERATOR'
TOKEN_LPAREN = 'LPAREN'
TOKEN_RPAREN = 'RPAREN'
TOKEN_EOF = 'EOF'

class Token:
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value
    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"

class Tokenizer:
    # Use word boundaries for functions to avoid partial matches (e.g., "sine")
    TOKEN_SPECS = [
        (r'(?:sin|cos|tan|sec|csc|cot|exp|sqrt)(?=\s*\()', TOKEN_FUNCTION),
        (r'\d+\.?\d*|\.\d+', TOKEN_NUMBER),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TOKEN_SYMBOL),
        (r'[\+\-\*\/^]', TOKEN_OPERATOR),
        (r'\(', TOKEN_LPAREN),
        (r'\)', TOKEN_RPAREN),
        (r'\s+', None),
    ]
    _COMPILED_SPECS = [(re.compile(pattern), ttype) for pattern, ttype in TOKEN_SPECS]

    def __init__(self, expression_string: str):
        self.expression_string = expression_string
        self.tokens = self._tokenize()
        self.current_token_index = 0

    def _tokenize(self) -> List[Token]:
        tokens = []
        pos = 0
        while pos < len(self.expression_string):
            match_found = False
            for regex, ttype in self._COMPILED_SPECS:
                match = regex.match(self.expression_string, pos)
                if match:
                    if ttype:
                        tokens.append(Token(ttype, match.group(0)))
                    pos = match.end()
                    match_found = True
                    break
            if not match_found:
                raise ValueError(f"Unexpected character at position {pos}: {self.expression_string[pos]}")
        tokens.append(Token(TOKEN_EOF, ''))
        return tokens
    
    def next(self) -> Token:
        if self.current_token_index < len(self.tokens):
            token = self.tokens[self.current_token_index]
            self.current_token_index += 1
            return token
        return Token(TOKEN_EOF, '')

class Parser:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.current_token = self.tokenizer.next()
        # Dictionary to store unique DAGNodes for canonical representation
        self.canonical_nodes: Dict[Tuple, DAGNode] = {}
        
    def _eat(self, token_type: str):
        if self.current_token.type == token_type:
            self.current_token = self.tokenizer.next()
        else:
            raise ValueError(f"Expected {token_type}, got {self.current_token.type} ('{self.current_token.value}')")

    def _create_node(self, value, children: Tuple[DAGNode, ...] = tuple()) -> DAGNode:
        key = (value, children)
        if key in self.canonical_nodes:
            return self.canonical_nodes[key]
        
        node = DAGNode(value, children)
        self.canonical_nodes[key] = node
        return node

    def parse(self) -> DAGNode:
        result = self._expr()
        if self.current_token.type != TOKEN_EOF:
            raise ValueError("Unexpected token at end of expression")
        return result

    def _expr(self):  # Handles + and -
        node = self._term()
        while self.current_token.type == TOKEN_OPERATOR and self.current_token.value in ('+', '-'):
            op = self.current_token.value
            self._eat(TOKEN_OPERATOR)
            right = self._term()
            node = self._create_node(op, (node, right))
        return node

    def _term(self):  # Handles * and /
        node = self._factor()
        while self.current_token.type == TOKEN_OPERATOR and self.current_token.value in ('*', '/'):
            op = self.current_token.value
            self._eat(TOKEN_OPERATOR)
            right = self._factor()
            node = self._create_node(op, (node, right))
        return node

    def _factor(self):  # Handles ^
        node = self._atom()
        if self.current_token.type == TOKEN_OPERATOR and self.current_token.value == '^':
            self._eat(TOKEN_OPERATOR)
            right = self._factor() 
            node = self._create_node('^', (node, right))
        return node
    
    def _atom(self):
        token = self.current_token
        if token.type == TOKEN_NUMBER:
            self._eat(TOKEN_NUMBER)
            return self._create_node(float(token.value))
        elif token.type == TOKEN_SYMBOL:
            self._eat(TOKEN_SYMBOL)
            return self._create_node(token.value)
        elif token.type == TOKEN_FUNCTION:
            func_name = token.value
            self._eat(TOKEN_FUNCTION)
            # Expect parentheses after function name
            if self.current_token.type != TOKEN_LPAREN:
                raise ValueError(f"Expected '(' after function '{func_name}'")
            self._eat(TOKEN_LPAREN)
            arg = self._expr()
            self._eat(TOKEN_RPAREN)
            return self._create_node(func_name, (arg,))
        elif token.type == TOKEN_LPAREN:
            self._eat(TOKEN_LPAREN)
            node = self._expr()
            self._eat(TOKEN_RPAREN)
            return node
        elif token.type == TOKEN_OPERATOR and token.value == '-':
            self._eat(TOKEN_OPERATOR)
            # Unary minus is represented as multiplication by -1
            neg_one = self._create_node(-1.0)
            factor = self._factor()
            return self._create_node('*', (neg_one, factor))
        
        raise ValueError(f"Invalid Expression")

# --- Helper and Formatting Functions ---

def to_latex(node: DAGNode):
    if not node.children:
        if isinstance(node.value, float):
            return str(int(node.value)) if node.value.is_integer() else str(node.value)
        return str(node.value)

    op = node.value
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    def format_child(child_node, is_left_child=False):
        child_latex = to_latex(child_node)
        child_op = child_node.value
        
        if not child_node.children: return child_latex
        
        op_prec = precedence.get(op, 99)
        child_prec = precedence.get(child_op, 99)
        
        needs_parens = False
        if child_prec < op_prec: 
            needs_parens = True
        elif child_prec == op_prec:
            if op == '^' and is_left_child: needs_parens = True # Power is right-associative
            if op in "+-*/" and not is_left_child: needs_parens = True # Left-associativity rule
        
        return f"({child_latex})" if needs_parens else child_latex

    args_latex = [format_child(c, i==0) for i, c in enumerate(node.children)]
    
    # Binary Operators
    if op == '+': return f"{args_latex[0]} + {args_latex[1]}"
    if op == '-': return f"{args_latex[0]} - {args_latex[1]}"
    
    if op == '*':
        # Check for unary minus: -1.0 * X should be formatted as -X
        if isinstance(node.children[0].value, float) and abs(node.children[0].value + 1.0) < 1e-9 and not node.children[0].children:
             return f"-{args_latex[1]}"

        # Check for 1 * X or X * 1 (should have been simplified, but for safety)
        if isinstance(node.children[0].value, float) and abs(node.children[0].value - 1.0) < 1e-9 and not node.children[0].children:
             return args_latex[1]
        if isinstance(node.children[1].value, float) and abs(node.children[1].value - 1.0) < 1e-9 and not node.children[1].children:
             return f"(\\cdot {args_latex[0]})"

        # Implicit multiplication for non-constants (2x or cos(x) 2x)
        if isinstance(node.children[0].value, float) and not node.children[0].children:
            return f"{args_latex[0]}{args_latex[1]}"
        
        # Use implicit multiplication by default (a space) for other terms
        return f"{args_latex[0]} \\cdot {args_latex[1]}"
        
    if op == '/': return f"\\frac{{{to_latex(node.children[0])}}}{{{to_latex(node.children[1])}}}"
    
    if op == '^': 
        base_latex = format_child(node.children[0], is_left_child=True)
        # Exponents are always wrapped in braces
        return f"{base_latex}^{{{args_latex[1]}}}"
    
    # Functions (Unary operators)
    if op in SUPPORTED_FUNCTIONS: return f"\\{op}({args_latex[0]})"
    
    return f"{op}({', '.join(args_latex)})"

def depends_on(node: DAGNode, var: str):
    if node.value == var:
        return True
    return any(depends_on(child, var) for child in node.children)


# --- Expression Simplifier ---
class Simplifier:
    def __init__(self, canonical_nodes: Dict[Tuple, DAGNode]):
        self.memo: Dict[DAGNode, DAGNode] = {}
        self.canonical_nodes = canonical_nodes
        self.zero = self._create_node(0.0)
        self.one = self._create_node(1.0)
        self.neg_one = self._create_node(-1.0)

    def _create_node(self, value, children: Tuple[DAGNode, ...] = tuple()) -> DAGNode:
        key = (value, children)
        if key in self.canonical_nodes:
            return self.canonical_nodes[key]
        
        node = DAGNode(value, children)
        self.canonical_nodes[key] = node
        return node
    
    def run(self, node: DAGNode) -> DAGNode:
        if node in self.memo:
            return self.memo[node]

        # If leaf node, nothing to simplify
        if not node.children:
            self.memo[node] = node
            return node

        # 1. Recursively simplify children
        simplified_children = tuple(self.run(child) for child in node.children)
        op = node.value
        result_node = None
        
        # Safely unpack 2 children (many ops are binary)
        left = simplified_children[0] if len(simplified_children) > 0 else None
        right = simplified_children[1] if len(simplified_children) > 1 else None
        
        # Helper for float comparison
        def is_float_equal(n: Optional[DAGNode], target_value: float) -> bool:
            return (n is not None) and isinstance(n.value, float) and abs(n.value - target_value) < 1e-9 and not n.children

        # --- Simplification Rules ---
        if op == '+':
            if left is not None and right is not None:
                if is_float_equal(left, 0.0):
                    result_node = right
                elif is_float_equal(right, 0.0):
                    result_node = left
                elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children:
                    result_node = self._create_node(left.value + right.value)

        elif op == '-':
            if left is not None and right is not None:
                if is_float_equal(right, 0.0):
                    result_node = left
                elif left == right:
                    result_node = self.zero
                elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children:
                    result_node = self._create_node(left.value - right.value)

        elif op == '*':
            if left is not None and right is not None:
                if is_float_equal(left, 0.0) or is_float_equal(right, 0.0):
                    result_node = self.zero
                elif is_float_equal(left, 1.0):
                    result_node = right
                elif is_float_equal(right, 1.0):
                    result_node = left
                elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children:
                    result_node = self._create_node(left.value * right.value)
                
                else:
                    def is_constant(n):
                        return isinstance(n.value, (float, int)) and not n.children

                    # Pattern: C1 * (C2 * X)  ->  (C1 * C2) * X
                    if is_constant(left) and right.value == '*' and is_constant(right.children[0]):
                        new_const_val = left.value * right.children[0].value
                        # Rebuild using the canonical node creator and simplify recursively
                        new_node = self._create_node('*', (self._create_node(new_const_val), right.children[1]))
                        result_node = self.run(new_node)
                    
                    # Pattern: (X * C1) * C2  ->  X * (C1 * C2)
                    elif is_constant(right) and left.value == '*' and is_constant(left.children[1]):
                        new_const_val = right.value * left.children[1].value
                        new_node = self._create_node('*', (left.children[0], self._create_node(new_const_val)))
                        result_node = self.run(new_node)

        elif op == '/':
            if left is not None and right is not None:
                if is_float_equal(left, 0.0) and not is_float_equal(right, 0.0):
                    result_node = self.zero
                elif is_float_equal(right, 1.0):
                    result_node = left
                elif left == right:
                    result_node = self.one
                elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children and not is_float_equal(right, 0.0):
                    result_node = self._create_node(left.value / right.value)

        elif op == '^':
            # Expect exactly two children for power
            if left is not None and right is not None:
                if is_float_equal(right, 1.0):
                    result_node = left
                elif is_float_equal(right, 0.0):
                    result_node = self.one
                elif is_float_equal(left, 1.0):
                    result_node = self.one
                elif is_float_equal(left, 0.0):
                    result_node = self.zero
                elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children:
                    # numeric power folding, careful with negative/zero exponents but allow general float pow
                    result_node = self._create_node(left.value ** right.value)

        # If no simplification rule was hit, rebuild the node from simplified children
        if result_node is None:
            result_node = self._create_node(op, simplified_children)

        # Cache the result and return
        self.memo[node] = result_node
        return result_node

# --- Derivative Computation with Memoization (DAG advantage) ---
class Differentiator:
    def __init__(self, variable: str, canonical_nodes: Dict[Tuple, DAGNode]):
        self.variable = variable
        self.steps = []
        self.memo: Dict[DAGNode, DAGNode] = {}
        self.canonical_nodes = canonical_nodes
        
        # Pre-create canonical constants for easy rule generation
        self.zero = self._create_node(0.0)
        self.one = self._create_node(1.0)
        self.neg_one = self._create_node(-1.0)

    def _create_node(self, value, children: Tuple[DAGNode, ...] = tuple()) -> DAGNode:
        key = (value, children)
        if key in self.canonical_nodes:
            return self.canonical_nodes[key]
        
        node = DAGNode(value, children)
        self.canonical_nodes[key] = node
        return node

    def _add_step(self, node: DAGNode, rule_key: str, explanation: str, prefix: str = "= "):
        self.steps.append({
            "id": f"step_{len(self.steps)}_{rule_key}",
            "prefix": prefix,
            # Convert DAGNode to LaTeX string for display
            "parts": [{"latex": to_latex(node), "explanation_key": rule_key}],
            "explanation_text": explanation
        })

    def run(self, node: DAGNode):
        self._add_step(node, "initial_expression", "Differentiating the expression:", prefix=f"\\frac{{d}}{{d{self.variable}}}")
        return self._differentiate(node)

    def _differentiate(self, node: DAGNode) -> DAGNode:
        if node in self.memo:
            return self.memo[node]

        # --- Base cases ---
        if not node.children:
            if node.value == self.variable:
                self._add_step(node, "variableRule", f"The derivative of {self.variable} is 1.")
                result = self.one
            elif isinstance(node.value, (int, float)) or (isinstance(node.value, str) and node.value not in SUPPORTED_FUNCTIONS):
                self._add_step(node, "constantRule", f"The derivative of a constant is 0.")
                result = self.zero
            else:
                # This should not happen if the parser is correct
                raise ValueError(f"Unhandled base case for value: {node.value}")
            
            self.memo[node] = result
            return result
        
        op = node.value
        args = node.children
        result_node = None

        # Standardized wording helper
        def sum_rule_start(n): self._add_step(n, "sumRule_start", "Applying the sum/difference rule:")
        def sum_rule_result(n): self._add_step(n, "sumRule_result", "Result of the sum/difference rule.")
        def product_rule_start(n): self._add_step(n, "productRule_start", "Applying the product rule:")
        def product_rule_result(n): self._add_step(n, "productRule_result", "Result of the product rule.")
        def quotient_rule_start(n): self._add_step(n, "quotientRule_start", "Applying the quotient rule:")
        def quotient_rule_result(n): self._add_step(n, "quotientRule_result", "Result of the quotient rule.")
        def power_rule_start(n): self._add_step(n, "powerRule_start", "Applying the power rule:")
        def power_rule_result(n): self._add_step(n, "powerRule_result", "Result of the power rule.")
        def chain_rule_start(n, rule_name): self._add_step(n, f"{rule_name}Rule_start", f"Applying the chain rule for {rule_name}:")
        def chain_rule_result(n, rule_name): self._add_step(n, f"{rule_name}Rule_result", "Result of the chain rule.")

        # --- Rule Implementations ---
        
        # Sum/Difference Rule
        if op in ('+', '-'):
            sum_rule_start(node)
            d_args = tuple(self._differentiate(arg) for arg in args)
            result_node = self._create_node(op, d_args)
            sum_rule_result(result_node)
        
        # Product Rule
        elif op == '*':
            u, v = args
            product_rule_start(node)
            du = self._differentiate(u)
            dv = self._differentiate(v)
            term1 = self._create_node('*', (du, v)) # u'v
            term2 = self._create_node('*', (u, dv)) # uv'
            result_node = self._create_node('+', (term1, term2))
            product_rule_result(result_node)
        
        # Quotient Rule
        elif op == '/':
            u, v = args
            quotient_rule_start(node)
            du = self._differentiate(u)
            dv = self._differentiate(v)
            num_term1 = self._create_node('*', (du, v)) # u'v
            num_term2 = self._create_node('*', (u, dv)) # uv'
            numerator = self._create_node('-', (num_term1, num_term2)) # u'v - uv'
            denominator = self._create_node('^', (v, self._create_node(2.0))) # v^2
            result_node = self._create_node('/', (numerator, denominator))
            quotient_rule_result(result_node)

        # Power Rule (f(x)^c) where exponent is constant
        elif op == '^':
            base, exp = args
            if not depends_on(exp, self.variable) and isinstance(exp.value, float):
                power_rule_start(node)
                
                exp_val = exp.value
                new_exp = self._create_node(exp_val - 1.0) 
                
                term1 = self._create_node('*', (exp, self._create_node('^', (base, new_exp)))) # c * u^(c-1)
                du = self._differentiate(base) # u'
                result_node = self._create_node('*', (term1, du)) # (c * u^(c-1)) * u'
                power_rule_result(result_node)
            else:
                raise NotImplementedError("Derivative of f(x)^g(x) is not implemented")

        # --- Chain Rule: functions ---
        elif op in SUPPORTED_FUNCTIONS:
            u = args[0]
            du = self._differentiate(u)

            # Generic chain rule helper that uses the standardized messages
            def apply_chain_rule(rule_name, inner_derivative_node):
                chain_rule_start(node, rule_name)
                # Multiply inner derivative (f'(u)) by u'
                result = self._create_node('*', (inner_derivative_node, du))
                chain_rule_result(result, rule_name)
                return result

            if op == 'sin':
                inner = self._create_node('cos', (u,))
                result_node = apply_chain_rule("sin", inner)
            elif op == 'cos':
                inner = self._create_node('*', (self.neg_one, self._create_node('sin', (u,))))
                result_node = apply_chain_rule("cos", inner)
            elif op == 'tan':
                inner = self._create_node('^', (self._create_node('sec', (u,)), self._create_node(2.0)))
                result_node = apply_chain_rule("tan", inner)
            elif op == 'sec':
                inner = self._create_node('*', (self._create_node('sec', (u,)), self._create_node('tan', (u,))))
                result_node = apply_chain_rule("sec", inner)
            elif op == 'csc':
                inner = self._create_node('*', (self.neg_one, self._create_node('*', (self._create_node('csc', (u,)), self._create_node('cot', (u,)))))) 
                result_node = apply_chain_rule("csc", inner)
            elif op == 'cot':
                inner = self._create_node('*', (self.neg_one, self._create_node('^', (self._create_node('csc', (u,)), self._create_node(2.0)))))
                result_node = apply_chain_rule("cot", inner)
            elif op == 'exp':
                inner = self._create_node('exp', (u,))
                result_node = apply_chain_rule("exp", inner)
            elif op == 'sqrt':
                two_sqrt_u = self._create_node('*', (self._create_node(2.0), self._create_node('sqrt', (u,))))
                inner = self._create_node('/', (self.one, two_sqrt_u))
                result_node = apply_chain_rule("sqrt", inner)
            else:
                raise ValueError(f"Differentiation rule for function '{op}' not implemented")
        else:
            raise ValueError(f"Differentiation rule for operator '{op}' not implemented")

        # Cache the result and return
        self.memo[node] = result_node
        return result_node

# --- Main Compute Function ---
def compute_derivative_dag(expression_str: str, variable_str: str):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # This dictionary holds all unique nodes, enforcing the DAG structure
    canonical_nodes: Dict[Tuple, DAGNode] = {}
    
    try:
        # 1. Tokenize and Parse into DAG
        tokenizer = Tokenizer(expression_str)
        parser = Parser(tokenizer)

        expression_dag = parser.parse()
        canonical_nodes.update(parser.canonical_nodes)

        # 2. Differentiate with step tracking (Memoization uses the DAG structure)
        differentiator = Differentiator(variable_str, canonical_nodes)
        derivative_dag = differentiator.run(expression_dag)
        
        # 3. Simplify the result before displaying
        def simplify_until_stable(node: DAGNode) -> DAGNode:
            current_node = node
            simplifier = Simplifier(canonical_nodes)
            for i in range(10): # Max 10 passes so more chance to converge
                simplified_node = simplifier.run(current_node)
                if simplified_node == current_node:
                    break
                current_node = simplified_node
            return current_node

        simplified_dag = simplify_until_stable(derivative_dag)
        
        # 4. Format final result
        derivative_latex = to_latex(simplified_dag)
        differentiator._add_step(simplified_dag, "final_derivative", "The final derivative is:", prefix="\\text{Simplified Result:} & = ")
        steps = differentiator.steps

    except Exception as e:
        logger.error(f"Error computing derivative for '{expression_str}': {e}", exc_info=True)
        steps = [{"id": "error", "prefix": "Error:", "parts": [], "explanation_text": str(e)}]
        derivative_latex = f"\\text{{Error: {str(e)}}}"

    finally:
        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        "derivative_latex": derivative_latex,
        "steps": steps,
        "execution_time_ms": (end_time - start_time) * 1000,
        "peak_memory_bytes": peak_memory,
    }
