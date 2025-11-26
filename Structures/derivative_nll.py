import re
import time
import tracemalloc
import logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Data Structure: Nested List-Like (NLL) ---
# The core data structure is a standard Python list, used recursively.
# - A number is a float: 3.14
# - A variable is a string: 'x'
# - An operation is a list: ['operator', operand1, operand2, ...]
# - Example: 2 * sin(x) -> ['*', 2.0, ['sin', 'x']]

# --- Tokenizer: Breaking the Expression into Tokens ---
TOKEN_NUMBER = 'NUMBER'
TOKEN_SYMBOL = 'SYMBOL'
TOKEN_FUNCTION = 'FUNCTION'
TOKEN_OPERATOR = 'OPERATOR'
TOKEN_LPAREN = 'LPAREN'
TOKEN_RPAREN = 'RPAREN'
TOKEN_EOF = 'EOF'

SUPPORTED_FUNCTIONS = {"sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "sqrt"}

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"

class Tokenizer:
    TOKEN_SPECS = [
        (r'\d+\.?\d*|\.\d+', TOKEN_NUMBER),
        (r'(?:sin|cos|tan|sec|csc|cot|exp|log|sqrt)(?=\s*\()', TOKEN_FUNCTION),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TOKEN_SYMBOL),
        (r'[\+\-\*\/^]', TOKEN_OPERATOR),
        (r'\(', TOKEN_LPAREN),
        (r'\)', TOKEN_RPAREN),
        (r'\s+', None),  # Skip whitespace
    ]
    _COMPILED_SPECS = [(re.compile(pattern), ttype) for pattern, ttype in TOKEN_SPECS]

    def __init__(self, text):
        self.text = text
        self.tokens = self._tokenize()
        self.index = 0

    def _tokenize(self):
        tokens = []
        pos = 0
        while pos < len(self.text):
            match_found = False
            for regex, ttype in self._COMPILED_SPECS:
                match = regex.match(self.text, pos)
                if match:
                    if ttype:
                        tokens.append(Token(ttype, match.group(0)))
                    pos = match.end()
                    match_found = True
                    break
            if not match_found:
                raise ValueError(f"Unexpected character at position {pos}: '{self.text[pos]}'")
        tokens.append(Token(TOKEN_EOF, ""))
        return tokens

    def next(self):
        if self.index < len(self.tokens):
            token = self.tokens[self.index]
            self.index += 1
            return token
        return Token(TOKEN_EOF, "")

# --- Parser: Building the NLL from Tokens ---
class Parser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.current_token = self.tokenizer.next()

    def _eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.tokenizer.next()
        else:
            raise ValueError(f"Parser Error: Expected {token_type}, but got {self.current_token.type}")

    def parse(self):
        result = self._expr()
        if self.current_token.type != TOKEN_EOF:
            raise ValueError("Parser Error: Unexpected token at end of expression")
        return result

    def _expr(self):  # Handles Addition (+) and Subtraction (-)
        node = self._term()
        while self.current_token.type == TOKEN_OPERATOR and self.current_token.value in ('+', '-'):
            op = self.current_token.value
            self._eat(TOKEN_OPERATOR)
            right = self._term()
            node = [op, node, right]
        return node

    def _term(self):  # Handles Multiplication (*) and Division (/)
        node = self._factor()
        while self.current_token.type == TOKEN_OPERATOR and self.current_token.value in ('*', '/'):
            op = self.current_token.value
            self._eat(TOKEN_OPERATOR)
            right = self._factor()
            node = [op, node, right]
        return node

    def _factor(self):  # Handles exponentiation (^)
        node = self._atom()
        if self.current_token.type == TOKEN_OPERATOR and self.current_token.value == '^':
            self._eat(TOKEN_OPERATOR)
            right = self._factor()  # Right-associativity
            node = ['^', node, right]
        return node

    def _atom(self):
        token = self.current_token
        if token.type == TOKEN_NUMBER:
            self._eat(TOKEN_NUMBER)
            return float(token.value)
        elif token.type == TOKEN_SYMBOL:
            self._eat(TOKEN_SYMBOL)
            return token.value
        elif token.type == TOKEN_FUNCTION:
            func_name = token.value
            self._eat(TOKEN_FUNCTION)
            self._eat(TOKEN_LPAREN)
            arg = self._expr()
            self._eat(TOKEN_RPAREN)
            return [func_name, arg]
        elif token.type == TOKEN_LPAREN:
            self._eat(TOKEN_LPAREN)
            node = self._expr()
            self._eat(TOKEN_RPAREN)
            return node
        elif token.type == TOKEN_OPERATOR and token.value == '-':
            self._eat(TOKEN_OPERATOR)
            return ['*', -1.0, self._factor()]
        raise ValueError(f"Invalid Expression")

# --- Helper and Formatting Functions ---
def to_latex(nll):
    if not isinstance(nll, list):
        if isinstance(nll, float):
            return str(int(nll)) if nll.is_integer() else str(nll)
        return str(nll)

    op = nll[0]
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    def format_child(child_nll, is_left_child=False):
        child_latex = to_latex(child_nll)
        if not isinstance(child_nll, list):
            return child_latex

        child_op = child_nll[0]
        op_prec = precedence.get(op, 99)
        child_prec = precedence.get(child_op, 99)

        if child_prec < op_prec or \
           (child_prec == op_prec and ((op == '^' and is_left_child) or (op in "+-*/" and not is_left_child))):
            return f"({child_latex})"
        return child_latex

    args_latex = [format_child(c, i == 0) for i, c in enumerate(nll[1:])]

    if op == '+': return f"{args_latex[0]} + {args_latex[1]}"
    if op == '-': return f"{args_latex[0]} - {args_latex[1]}"
    if op == '*':
        if nll[1] == -1.0: return f"-{args_latex[1]}"
        
        # Rule 2: Use implicit multiplication for a number and a variable/function (e.g., 4x)
        if isinstance(nll[1], float) and not isinstance(nll[2], float): return f"{to_latex(nll[1])}{args_latex[1]}"

        # Rule 3 (Default): For all other cases (x*y, x*4, 4*1), use \cdot
        return f"{args_latex[0]} \\cdot {args_latex[1]}"
    if op == '/': return f"\\frac{{{to_latex(nll[1])}}}{{{to_latex(nll[2])}}}"
    if op == '^': return f"{{{args_latex[0]}}}^{{{args_latex[1]}}}"
    if op in SUPPORTED_FUNCTIONS: return f"\\{op}({args_latex[0]})"
    return f"{op}({', '.join(args_latex)})"

def depends_on(nll, var):
    if nll == var:
        return True
    if isinstance(nll, list):
        return any(depends_on(child, var) for child in nll[1:])
    return False

# --- Expression Simplifier ---
class Simplifier:
    def __init__(self):
        self.memo = {}

    def _to_tuple(self, nll):
        """Recursively converts a nested list to a nested tuple for hashing."""
        if isinstance(nll, list):
            return tuple(self._to_tuple(item) for item in nll)
        return nll

    def run(self, nll):
        # Use a hashable tuple version as the key for memoization
        key = self._to_tuple(nll)
        if key in self.memo:
            return self.memo[key]
        
        if not isinstance(nll, list):
            return nll

        simplified_children = [self.run(child) for child in nll[1:]]
        op = nll[0]
        result_nll = None

        if op in ('+', '-', '*', '/'):
            left, right = simplified_children
            if op == '+':
                if left == 0.0: result_nll = right
                elif right == 0.0: result_nll = left
                elif isinstance(left, float) and isinstance(right, float): result_nll = left + right
            elif op == '-':
                if right == 0.0: result_nll = left
                elif left == right: result_nll = 0.0
                elif isinstance(left, float) and isinstance(right, float): result_nll = left - right
            elif op == '*':
                if left == 0.0 or right == 0.0: result_nll = 0.0
                elif left == 1.0: result_nll = right
                elif right == 1.0: result_nll = left
                # Associativity rule: a * (b * c) -> (a * b) * c
                elif isinstance(left, float) and isinstance(right, list) and right[0] == '*' and isinstance(right[1], float):
                    new_const = left * right[1]
                    result_nll = ['*', new_const, right[2]]
                # Associativity rule: (a * b) * c -> (a * c) * b
                elif isinstance(right, float) and isinstance(left, list) and left[0] == '*' and isinstance(left[1], float):
                    new_const = right * left[1]
                    result_nll = ['*', new_const, left[2]]
                elif isinstance(left, float) and isinstance(right, float): result_nll = left * right
            elif op == '/':
                if left == 0.0: result_nll = 0.0
                elif right == 1.0: result_nll = left
                elif left == right and left != 0.0: result_nll = 1.0
                elif isinstance(left, float) and isinstance(right, float) and right != 0.0: result_nll = left / right

        elif op == '^':
            base, exp = simplified_children
            if exp == 1.0: result_nll = base
            elif exp == 0.0: result_nll = 1.0
            elif base == 1.0: result_nll = 1.0
            elif base == 0.0: result_nll = 0.0

        if result_nll is None:
            result_nll = [op] + simplified_children
        
        # Store the result (a list) in the memo using the tuple key
        self.memo[key] = result_nll
        return result_nll

# --- Derivative Computation with Step-by-Step Logging ---
class Differentiator:
    def __init__(self, variable):
        self.variable = variable
        self.steps = []

    def _add_step(self, nll, rule_key, explanation, prefix="= "):
        self.steps.append({
            "id": f"step_{len(self.steps)}_{rule_key}",
            "prefix": prefix,
            "parts": [{"latex": to_latex(nll), "explanation_key": rule_key}],
            "explanation_text": explanation
        })

    def run(self, nll):
        self._add_step(nll, "initial_expression", "Differentiating the expression:", prefix=f"\\frac{{d}}{{d{self.variable}}}")
        return self._differentiate(nll)

    def _apply_chain_rule(self, nll, op, result_func):
        u = nll[1]
        rule_name = op.capitalize() + " Rule"
        self._add_step(nll, f"{op}Rule_start", f"Applying the Chain Rule for {op.capitalize()}:")
        du = self._differentiate(u)
        result_nll = result_func(u, du)
        self._add_step(result_nll, f"{op}Rule_result", f"Result of the {rule_name}.")
        return result_nll

    def _differentiate(self, nll):
        if not isinstance(nll, list):
            if nll == self.variable:
                self._add_step(nll, "variableRule", f"The derivative of {self.variable} is 1.")
                return 1.0
            if isinstance(nll, (int, float, str)):
                self._add_step(nll, "constantRule", f"The derivative of a constant is 0.")
                return 0.0
            return nll

        op = nll[0]
        args = nll[1:]

        if op in ('+', '-'):
            self._add_step(nll, "sumRule_start", "Applying the Sum/Difference Rule.")
            d_args = [self._differentiate(arg) for arg in args]
            result_nll = [op] + d_args
            self._add_step(result_nll, "sumRule_result", "Result of the Sum/Difference Rule.")
            return result_nll

        if op == '*':
            u, v = args
            self._add_step(nll, "productRule_start", "Applying the Product Rule: ")
            du = self._differentiate(u)
            dv = self._differentiate(v)
            result_nll = ['+', ['*', du, v], ['*', u, dv]]
            self._add_step(result_nll, "productRule_result", "Result of the Product Rule.")
            return result_nll

        if op == '/':
            u, v = args
            self._add_step(nll, "quotientRule_start", "Applying the Quotient Rule: ")
            du = self._differentiate(u)
            dv = self._differentiate(v)
            num = ['-', ['*', du, v], ['*', u, dv]]
            den = ['^', v, 2.0]
            result_nll = ['/', num, den]
            self._add_step(result_nll, "quotientRule_result", "Result of the Quotient Rule.")
            return result_nll

        if op == '^':
            base, exp = args
            if not depends_on(exp, self.variable):
                self._add_step(nll, "powerRule_start", "Applying the Power Rule: ")
                du = self._differentiate(base)
                new_exp = exp - 1.0
                term1 = ['*', exp, ['^', base, new_exp]]
                result_nll = ['*', term1, du]
                self._add_step(result_nll, "powerRule_result", "Result of the Power Rule.")
                return result_nll
            else:
                raise NotImplementedError("Derivative of f(x)^g(x) is not implemented")

        if op in SUPPORTED_FUNCTIONS:
            if op == 'sin': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['cos', u], du])
            if op == 'cos': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['*', -1.0, ['sin', u]], du])
            if op == 'tan': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['^', ['sec', u], 2.0], du])
            if op == 'sec': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['*', ['sec', u], ['tan', u]], du])
            if op == 'csc': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['*', -1.0, ['csc', u]], ['*', ['cot', u], du]])
            if op == 'cot': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['*', -1.0, ['^', ['csc', u], 2.0]], du])
            if op == 'exp': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['exp', u], du])
            if op == 'log': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['/', 1.0, u], du])
            if op == 'sqrt': return self._apply_chain_rule(nll, op, lambda u, du: ['*', ['/', 1.0, ['*', 2.0, nll]], du])
        
        raise ValueError(f"Differentiation rule for '{op}' not implemented")

# --- Main Compute Function ---
def compute_derivative_nll(expression_str, variable_str):
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        tokenizer = Tokenizer(expression_str)
        parser = Parser(tokenizer)
        expression_nll = parser.parse()

        differentiator = Differentiator(variable_str)
        derivative_nll = differentiator.run(expression_nll)

        simplifier = Simplifier()
        simplified_nll = simplifier.run(derivative_nll)

        steps = differentiator.steps
        derivative_latex = to_latex(simplified_nll)
        steps.append({
            "id": "final_derivative",
            "prefix": "= ",
            "parts": [{"latex": derivative_latex, "explanation_key": "final_derivative"}],
            "explanation_text": "The final derivative is:"
        })

    except Exception as e:
        logger.error(f"Error computing NLL derivative for '{expression_str}': {e}", exc_info=True)
        derivative_latex = f"\\text{{Error: {str(e)}}}"
        steps = [{"id": "error", "prefix": "Error:", "parts": [], "explanation_text": str(e)}]
    
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


