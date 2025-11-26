import logging
import json
import asyncio
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
        "https://thesis-calculator-frontend-39m1yjfv8-raniels-projects-2ea24826.vercel.app/solver",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# -------------------------------------------------------------------
# Expression Validator (NO SYMPY)
# -------------------------------------------------------------------
def is_valid_expression(expr: str):
    if not expr or not expr.strip():
        return False, "Expression cannot be empty."

    invalid_sequences = ["++", "--", "**", "//", "^^", "+-", "-+", "()", "( )"]
    for seq in invalid_sequences:
        if seq in expr:
            return False, f"Invalid syntax sequence detected: '{seq}'."
        
    unsupported_functions = [
        "sinh", "cosh", "tanh", "coth", "sech", "csch",
        "arcsinh", "arccosh", "arctanh",
        "arcsin", "arccos", "arctan", "arcsec", "arccsc", "arccot",
        "log", "ln", "integrate", "diff" 
    ]

    for func in unsupported_functions:
        if func in expr:
            return False, f"Topic is outside of scope: {func}"
    
    stack = 0
    for ch in expr:
        if ch == '(':
            stack += 1
        elif ch == ')':
            stack -= 1
        if stack < 0:
            return False, "Unexpected closing parenthesis )"

    if stack != 0:
        return False, "Unbalanced parentheses in the expression."

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

    # Step 1 — Immediate Basic Validation
    is_valid, error_msg = is_valid_expression(expression)
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

                # Try to compute the derivative
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

                # After warmup → record metrics
                if run_index >= warmup_runs:
                    times.append(result_data['execution_time_ms'])
                    memories.append(result_data['peak_memory_bytes'])

                    if run_index == warmup_runs:
                        derivative_latex = result_data.get("derivative_latex", "")
                        steps = result_data.get("steps", [])

            # Compute averages
            avg_time = sum(times) / measured_runs
            avg_mem = sum(memories) / measured_runs

            final_results[ds] = {
                'derivative': derivative_latex,
                'steps': steps,
                'avgTime': avg_time,
                'avgMemory': avg_mem,
            }

        # Final message to frontend
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
    logger.debug(f"Solve request: {expression}")
    return StreamingResponse(
        benchmark_generator(expression, variable),
        media_type="text/event-stream"
    )


@app.post("/generate")
async def generate_expression_endpoint(input_data: GenerationInput):
    """
    NOTE: No SymPy used — your `generate_random_expression()` must return:
    - Python string expression
    - Or provide your own LaTeX generator (if needed)
    """
    try:
        expr, expr_latex = generate_random_expression(
            variables=input_data.variables,
            num_terms=input_data.num_terms,
            max_depth=input_data.max_depth
        )

        return {
            "expression_string": expr,
            "expression_latex": expr_latex
        }

    except Exception as e:
        logger.error("Generation error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
