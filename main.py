import logging
import json
import asyncio
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

# Your derivative engines (untouched)
from Structures.derivative_ast import compute_derivative_ast
from Structures.derivative_dag import compute_derivative_dag
from Structures.derivative_nll import compute_derivative_nll

# Your random expression generator
from generate_expression import generate_random_expression

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4000",
        "https://*.vercel.app",
        "https://thesis-calculator-frontend.vercel.app"        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Symbol Normalization (π → 3.14, √ → sqrt)
# -------------------------------------------------------------------
def normalize_expression(expr: str):
    if not expr:
        return expr
    expr = expr.replace("π", "3.14")
    expr = expr.replace("√", "sqrt")
    return expr

# -------------------------------------------------------------------
# Render Health Endpoints
# -------------------------------------------------------------------
@app.get("/ping")
async def ping():
    logger.info("🔔 Uptime ping received")
    return {"status": "ok", "message": "Backend is alive"}

@app.get("/uptime")
async def uptime():
    logger.info("🟢 UptimeRobot pinged this server.")
    return {"status": "alive"}

# -------------------------------------------------------------------
# Expression Validator (NO SYMPY)
# -------------------------------------------------------------------

ALLOWED_VARIABLES = {"x", "y", "z"}
ALLOWED_FUNCTIONS = {"sin", "cos", "tan", "sec", "csc", "cot", "sqrt", "exp"}

def is_valid_token(token: str):
    if token in ALLOWED_VARIABLES:
        return True
    if re.fullmatch(r"\d+(\.\d+)?", token):
        return True
    if token in ALLOWED_FUNCTIONS:
        return True
    return False

def validate_expression(expr: str):
    if not expr or not expr.strip():
        return False, "Expression cannot be empty."

    expr = expr.replace(" ", "")

    tokens = re.findall(r"[a-zA-Z]+|\d+\.\d+|\d+|[\+\-\*/\^\(\)]", expr)

    for token in tokens:

        if token.isalpha():
            if len(token) > 1 and token not in ALLOWED_FUNCTIONS:
                return False, f"Invalid token '{token}'. Unknown function or variable."
            if len(token) == 1 and token not in ALLOWED_VARIABLES:
                return False, f"Invalid variable '{token}'. Only x, y, z allowed."

            prefixes = ("s", "c", "t", "e")
            if token[0] in prefixes and token not in ALLOWED_FUNCTIONS and token not in ALLOWED_VARIABLES:
                return False, f"Unknown function '{token}'."

        if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", token):
            return False, f"Invalid token '{token}'. Variables must be letters only."

        if not is_valid_token(token) and not re.fullmatch(r"[\+\-\*/\^\(\)]", token):
            return False, f"Invalid token '{token}'."

    stack = 0
    for ch in expr:
        if ch == '(':
            stack += 1
        elif ch == ')':
            stack -= 1
        if stack < 0:
            return False, "Unexpected closing parenthesis."
    if stack != 0:
        return False, "Unbalanced parentheses."

    return True, None

# -------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------
class ExpressionInput(BaseModel):
    expression: str
    variable: str = 'x'


class GenerationInput(BaseModel):
    num_terms: Optional[int] = 3
    max_depth: Optional[int] = 2
    variables: Optional[List[str]] = ['x']

# -------------------------------------------------------------------
# Streaming Benchmark Engine
# -------------------------------------------------------------------
async def benchmark_generator(expression: str, variable: str):

    expression = normalize_expression(expression)

    is_valid, error_msg = validate_expression(expression)
    if not is_valid:
        error_response = {
            'type': 'error',
            'detail': error_msg
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        return

    data_structures = ['AST', 'DAG', 'NLL']
    total_runs = 30
    warmup_runs = 10
    measured_runs = total_runs - warmup_runs

    final_results = {
        ds: {'derivative': None, 'steps': [], 'avgTime': None, 'avgMemory': None}
        for ds in data_structures
    }

    try:
        for ds in data_structures:

            compute_func = {
                'AST': compute_derivative_ast,
                'DAG': compute_derivative_dag,
                'NLL': compute_derivative_nll
            }[ds]

            times = []
            memories = []
            derivative_latex = ""
            steps = []

            for run_index in range(total_runs):

                try:
                    result_data = compute_func(expression, variable)

                except NotImplementedError as nie:
                    err = {
                        'type': 'error',
                        'detail': f"{ds} structure does not implement this rule: {str(nie)}"
                    }
                    yield f"data: {json.dumps(err)}\n\n"
                    return

                except Exception as e:
                    err = {
                        'type': 'error',
                        'detail': f"{ds} calculation failed: {str(e)}"
                    }
                    yield f"data: {json.dumps(err)}\n\n"
                    return

                if run_index >= warmup_runs:
                    times.append(result_data['execution_time_ms'])
                    memories.append(result_data['peak_memory_bytes'])

                    if run_index == warmup_runs:
                        derivative_latex = result_data.get("derivative_latex", "")
                        steps = result_data.get("steps", [])

            avg_time = sum(times) / measured_runs
            avg_mem = sum(memories) / measured_runs

            final_results[ds] = {
                'derivative': derivative_latex,
                'steps': steps,
                'avgTime': avg_time,
                'avgMemory': avg_mem,
            }

        final_msg = {
            'type': 'complete',
            'results': final_results
        }
        yield f"data: {json.dumps(final_msg)}\n\n"

    except Exception as e:
        logger.error("Unexpected benchmark error", exc_info=True)
        err = {
            'type': 'error',
            'detail': f"Unexpected server error: {str(e)}"
        }
        yield f"data: {json.dumps(err)}\n\n"

# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------
@app.get("/solve_stream")
async def solve_derivative_stream(expression: str, variable: str = 'x'):

    expression = normalize_expression(expression)

    logger.debug(f"Solve request (normalized): {expression}")
    return StreamingResponse(
        benchmark_generator(expression, variable),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "https://thesis-calculator-frontend-39m1yjfv8-raniels-projects-2ea24826.vercel.app",
        }
    )

@app.post("/generate")
async def generate_expression_endpoint(input_data: GenerationInput):
    try:
        expr_sym, expr_str, expr_latex = generate_random_expression(
            variables=input_data.variables,
            num_terms=input_data.num_terms,
            max_depth=input_data.max_depth
        )

        return {
            "expression_string": expr_str,
            "expression_latex": expr_str 
        }

    except Exception as e:
        logger.error("Generation error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
