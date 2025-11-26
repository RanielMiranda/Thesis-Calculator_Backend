import re
import time
import tracemalloc
import logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Abstract Syntax Tree (AST) Node ---
class ASTNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def __repr__(self):
        if not self.children:
            return str(self.value)
        return f"{self.value}({', '.join(map(str, self.children))})"

# --- Tokenizer: Breaking the Expression into Tokens ---
TOKEN_NUMBER = 'NUMBER'
TOKEN_SYMBOL = 'SYMBOL'
TOKEN_FUNCTION = 'FUNCTION'
TOKEN_OPERATOR = 'OPERATOR'
TOKEN_LPAREN = 'LPAREN'
TOKEN_RPAREN = 'RPAREN'
TOKEN_EOF = 'EOF'

# A set of known mathematical functions.
SUPPORTED_FUNCTIONS = {"sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "sqrt"}

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"

class Tokenizer:
    # Function regex changed to use word boundaries so 'sine' doesn't match 'sin' + 'e'
    TOKEN_SPECS = [
        (r'\d+\.?\d*|\.\d+', TOKEN_NUMBER),
        (r'\b(?:sin|cos|tan|sec|csc|cot|exp|log|sqrt)\b', TOKEN_FUNCTION),
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

    def peek(self):
        if self.index < len(self.tokens):
            return self.tokens[self.index]
        return Token(TOKEN_EOF, "")

# --- Parser: Building the AST from Tokens ---
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
            node = ASTNode(op, [node, right])
        return node

    def _term(self):  # Handles Multiplication (*) and Division (/) and implicit multiplication
        node = self._factor()

        while True:
            # explicit * or /
            if self.current_token.type == TOKEN_OPERATOR and self.current_token.value in ('*', '/'):
                op = self.current_token.value
                self._eat(TOKEN_OPERATOR)
                right = self._factor()
                node = ASTNode(op, [node, right])
                continue

            # implicit multiplication
            if self.current_token.type in (TOKEN_NUMBER, TOKEN_SYMBOL, TOKEN_FUNCTION, TOKEN_LPAREN):
                right = self._factor()
                node = ASTNode('*', [node, right])
                continue

            break

        return node

    def _factor(self):  # Handles exponentiation (^)
        node = self._atom()
        if self.current_token.type == TOKEN_OPERATOR and self.current_token.value == '^':
            self._eat(TOKEN_OPERATOR)
            right = self._factor()
            node = ASTNode('^', [node, right])
        return node

    def _atom(self):
        token = self.current_token
        if token.type == TOKEN_NUMBER:
            self._eat(TOKEN_NUMBER)
            return ASTNode(float(token.value))
        elif token.type == TOKEN_SYMBOL:
            self._eat(TOKEN_SYMBOL)
            return ASTNode(token.value)
        elif token.type == TOKEN_FUNCTION:
            func_name = token.value
            self._eat(TOKEN_FUNCTION)
            # allow both f(x) and f x (e.g. sin x) - prefer explicit parentheses if present
            if self.current_token.type == TOKEN_LPAREN:
                self._eat(TOKEN_LPAREN)
                arg = self._expr()
                self._eat(TOKEN_RPAREN)
            else:
                # no parentheses: treat next atom as argument (e.g., "sin x" or "sin2")
                arg = self._atom()
            return ASTNode(func_name, [arg])
        elif token.type == TOKEN_LPAREN:
            self._eat(TOKEN_LPAREN)
            node = self._expr()
            self._eat(TOKEN_RPAREN)
            return node
        elif token.type == TOKEN_OPERATOR and token.value == '-':
            self._eat(TOKEN_OPERATOR)
            # Represent as multiplication by -1
            return ASTNode('*', [ASTNode(-1.0), self._factor()])

        raise ValueError(f"Invalid Expression")

# --- Helper and Formatting Functions ---
def to_latex(node):
    if not node.children:
        if isinstance(node.value, float):
            return str(int(node.value)) if node.value.is_integer() else str(node.value)
        return str(node.value)

    op = node.value

    # Define operator precedence for parenthesis insertion
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    def format_child(child_node, is_left_child=False):
        child_latex = to_latex(child_node)
        child_op = child_node.value

        if not child_node.children:
            return child_latex

        op_prec = precedence.get(op, 99)
        child_prec = precedence.get(child_op, 99)

        if child_prec < op_prec:
            return f"({child_latex})"
        if child_prec == op_prec:
            if op == '^' and is_left_child:
                return f"({child_latex})"
            if op in "+-*/" and not is_left_child:
                return f"({child_latex})"
        return child_latex

    args_latex = [format_child(c, i == 0) for i, c in enumerate(node.children)]

    if op == '+':
        return f"{args_latex[0]} + {args_latex[1]}"
    if op == '-':
        return f"{args_latex[0]} - {args_latex[1]}"
    if op == '*':
            left_child = node.children[0]
            right_child = node.children[1]

            is_left_const = isinstance(left_child.value, float) and not left_child.children
            is_right_const = isinstance(right_child.value, float) and not right_child.children

            # Rule 1: Handle unary minus like -x
            if is_left_const and left_child.value == -1.0:
                return f"-{args_latex[1]}"

            # Rule 2: Use implicit multiplication for a number and a variable/function (e.g., 4x)
            if is_left_const and not is_right_const:
                return f"{to_latex(left_child)}{args_latex[1]}"

            # Rule 3 (Default): For all other cases (x*y, x*4, 4*1), use \cdot
            return f"{args_latex[0]} \\cdot {args_latex[1]}"

    if op == '/':
        return f"\\frac{{{to_latex(node.children[0])}}}{{{to_latex(node.children[1])}}}"
    if op == '^':
        return f"{{{args_latex[0]}}}^{{{args_latex[1]}}}"
    if op in SUPPORTED_FUNCTIONS:
        return f"\\{op}({args_latex[0]})"

    return f"{op}({', '.join(args_latex)})"

def depends_on(node, var):
    if node.value == var:
        return True
    return any(depends_on(child, var) for child in node.children)

# --- Expression Simplifier ---
class Simplifier:
    def __init__(self):
        self.memo = {}

    def run(self, node):
        if id(node) in self.memo:
            return self.memo[id(node)]

        if not node.children:
            return node

        simplified_children = [self.run(child) for child in node.children]
        op = node.value
        result_node = None

        if op == '+':
            left, right = simplified_children
            if left.value == 0.0 and not left.children:
                result_node = right
            elif right.value == 0.0 and not right.children:
                result_node = left
            elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children:
                result_node = ASTNode(left.value + right.value)

        elif op == '-':
            left, right = simplified_children
            if right.value == 0.0 and not right.children:
                result_node = left
            elif left.value == right.value and not left.children and not right.children:
                result_node = ASTNode(0.0)
            elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children:
                result_node = ASTNode(left.value - right.value)

        elif op == '*':
            left, right = simplified_children
            if (left.value == 0.0 and not left.children) or (right.value == 0.0 and not right.children):
                result_node = ASTNode(0.0)
            elif left.value == 1.0 and not left.children:
                result_node = right
            elif right.value == 1.0 and not right.children:
                result_node = left
            elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children:
                result_node = ASTNode(left.value * right.value)
            
            else:
                def is_constant(n):
                    return isinstance(n.value, (float, int)) and not n.children

                # Pattern: C1 * (C2 * X)  ->  (C1 * C2) * X
                if is_constant(left) and right.value == '*' and is_constant(right.children[0]):
                    new_const_val = left.value * right.children[0].value
                    # Rebuild the node and simplify it again recursively
                    new_node = ASTNode('*', [ASTNode(new_const_val), right.children[1]])
                    result_node = self.run(new_node)
                
                # Pattern: (X * C1) * C2  ->  X * (C1 * C2)
                elif is_constant(right) and left.value == '*' and is_constant(left.children[1]):
                    new_const_val = right.value * left.children[1].value
                    new_node = ASTNode('*', [left.children[0], ASTNode(new_const_val)])
                    result_node = self.run(new_node)

        elif op == '/':
            left, right = simplified_children
            if left.value == 0.0 and not left.children:
                result_node = ASTNode(0.0)
            elif right.value == 1.0 and not right.children:
                result_node = left
            elif left.value == right.value and not left.children and not right.children and left.value != 0.0:
                result_node = ASTNode(1.0)
            elif isinstance(left.value, float) and isinstance(right.value, float) and not left.children and not right.children and right.value != 0.0:
                result_node = ASTNode(left.value / right.value)

        elif op == '^':
            base, exp = simplified_children

            if (isinstance(exp.value, (float, int)) and abs(exp.value - 1.0) < 1e-9) and not exp.children:
                result_node = base
            elif (isinstance(exp.value, (float, int)) and abs(exp.value - 0.0) < 1e-9) and not exp.children:
                result_node = ASTNode(1.0)
            elif (isinstance(base.value, (float, int)) and abs(base.value - 1.0) < 1e-9) and not base.children:
                result_node = ASTNode(1.0)
            elif (isinstance(base.value, (float, int)) and abs(base.value - 0.0) < 1e-9) and not base.children:
                result_node = ASTNode(0.0)

        if result_node is None:
            result_node = ASTNode(op, simplified_children)

        self.memo[id(node)] = result_node
        return result_node

# --- Derivative Computation with Step-by-Step Logging ---
class Differentiator:
    def __init__(self, variable):
        self.variable = variable
        self.steps = []

    def _add_step(self, node, rule_key, explanation, prefix="= "):
        self.steps.append({
            "id": f"step_{len(self.steps)}_{rule_key}",
            "prefix": prefix,
            "parts": [{"latex": to_latex(node), "explanation_key": rule_key}],
            "explanation_text": explanation
        })

    def run(self, node):
        self._add_step(node, "initial_expression", "Differentiating the expression:", prefix=f"\\frac{{d}}{{d{self.variable}}}")
        return self._differentiate(node)

    def _apply_chain_rule(self, node, op, result_func):
        u = node.children[0]
        rule_name = op.capitalize() + " Rule"
        self._add_step(node, f"{op}Rule_start", f"Applying the Chain Rule for {op.capitalize()}: ")
        du = self._differentiate(u)
        result_node = result_func(u, du)
        self._add_step(result_node, f"{op}Rule_result", f"Result of the {rule_name}.")
        return result_node

    def _differentiate(self, node):
        # Base cases: constants or the variable itself
        if not node.children:
            if node.value == self.variable:
                self._add_step(node, "variableRule", f"The derivative of {self.variable} is 1.")
                return ASTNode(1.0)
            if isinstance(node.value, (int, float, str)):
                self._add_step(node, "constantRule", f"The derivative of a constant is 0.")
                return ASTNode(0.0)
            return node

        op = node.value
        args = node.children

        if op in ('+', '-'):
            self._add_step(node, "sumRule_start", "Applying the Sum/Difference Rule.")
            d_args = [self._differentiate(arg) for arg in args]
            result_node = ASTNode(op, d_args)
            self._add_step(result_node, "sumRule_result", "Result of the Sum/Difference Rule.")
            return result_node

        if op == '*':
            u, v = args
            self._add_step(node, "productRule_start", "Applying the Product Rule: ")
            du = self._differentiate(u)
            dv = self._differentiate(v)
            result_node = ASTNode('+', [ASTNode('*', [du, v]), ASTNode('*', [u, dv])])
            self._add_step(result_node, "productRule_result", "Result of the Product Rule.")
            return result_node

        if op == '/':
            u, v = args
            self._add_step(node, "quotientRule_start", "Applying the Quotient Rule: ")
            du = self._differentiate(u)
            dv = self._differentiate(v)
            num = ASTNode('-', [ASTNode('*', [du, v]), ASTNode('*', [u, dv])])
            den = ASTNode('^', [v, ASTNode(2.0)])
            result_node = ASTNode('/', [num, den])
            self._add_step(result_node, "quotientRule_result", "Result of the Quotient Rule.")
            return result_node

        if op == '^':
            base, exp = args
            if not depends_on(exp, self.variable):  # Power Rule: f(x)^c
                self._add_step(node, "powerRule_start", "Applying the Power Rule: ")
                du = self._differentiate(base)
                new_exp_val = exp.value - 1.0 if isinstance(exp.value, float) else float(exp.value) - 1.0
                new_exp = ASTNode(new_exp_val)
                term1 = ASTNode('*', [exp, ASTNode('^', [base, new_exp])])
                result_node = ASTNode('*', [term1, du])
                self._add_step(result_node, "powerRule_result", "Result of the Power Rule.")
                return result_node
            else:
                raise NotImplementedError("Derivative of f(x)^g(x) is not implemented")
            
                         

        # Chain rule for functions
        if op in SUPPORTED_FUNCTIONS:
            if op == 'sin':
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('cos', [u]), du]))
            if op == 'cos':
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('*', [ASTNode(-1.0), ASTNode('sin', [u])]), du]))
            if op == 'tan':
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('^', [ASTNode('sec', [u]), ASTNode(2.0)]), du]))
            if op == 'sec':
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('*', [ASTNode('sec', [u]), ASTNode('tan', [u])]), du]))
            if op == 'csc':
                # -csc(u)cot(u) * u'
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('*', [ASTNode(-1.0), ASTNode('csc', [u])]), ASTNode('*', [ASTNode('cot', [u]), du])]))
            if op == 'cot':
                # -csc^2(u) * u'
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('*', [ASTNode(-1.0), ASTNode('^', [ASTNode('csc', [u]), ASTNode(2.0)])]), du]))
            if op == 'exp':
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('exp', [u]), du]))
            if op == 'log':
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('/', [ASTNode(1.0), u]), du]))
            if op == 'sqrt':
                return self._apply_chain_rule(node, op, lambda u, du: ASTNode('*', [ASTNode('/', [ASTNode(1.0), ASTNode('*', [ASTNode(2.0), node])]), du]))

        raise ValueError(f"Differentiation rule for '{op}' not implemented")

# --- Main Compute Function ---
def compute_derivative_ast(expression_str, variable_str):
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        # 1. Tokenize and Parse
        tokenizer = Tokenizer(expression_str)
        parser = Parser(tokenizer)
        expression_ast = parser.parse()

        # 2. Differentiate with step tracking
        differentiator = Differentiator(variable_str)
        derivative_ast = differentiator.run(expression_ast)

        # 3. Simplify the result before displaying
        simplifier = Simplifier()
        simplified_ast = simplifier.run(derivative_ast)

        steps = differentiator.steps

        # 4. Format final result
        derivative_latex = to_latex(simplified_ast)
        steps.append({
            "id": "final_derivative",
            "prefix": "= ",
            "parts": [{"latex": derivative_latex, "explanation_key": "final_derivative"}],
            "explanation_text": "The final derivative is:"
        })

    except Exception as e:
        logger.error(f"Error computing derivative for '{expression_str}': {e}", exc_info=True)
        end_time = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "derivative_latex": f"\\text{{Error: {str(e)}}}",
            "steps": [{"id": "error", "prefix": "Error:", "parts": [], "explanation_text": str(e)}],
            "execution_time_ms": (end_time - start_time) * 1000,
            "peak_memory_bytes": peak_memory,
        }
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
