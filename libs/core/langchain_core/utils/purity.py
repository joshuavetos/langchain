"""Security utilities for STTI-001 bytecode analysis."""

from __future__ import annotations

import dis
import inspect
from typing import Callable, Any

# Opcodes that allow external state access or manipulation
FORBIDDEN_OPCODES = {
    "IMPORT_NAME",
    "IMPORT_FROM",
    "IMPORT_STAR",
    "STORE_GLOBAL",
    "DELETE_GLOBAL",
}

# Attributes used in common Python "jailbreak" or escape patterns
FORBIDDEN_ATTRIBUTES = {
    "__subclasses__",
    "__builtins__",
    "__globals__",
    "__getattribute__",
    "__setattr__",
}

class PurityError(SecurityError):
    """Raised when a tool fails the STTI-001 Purity Gate analysis."""

def analyze_bytecode_purity(func: Callable) -> bool:
    """
    Scans the bytecode of a function to ensure it cannot escape its sandbox.
    Enforces the 'No Side Effect Without Provenance' invariant at the CPU level.
    """
    if not inspect.isfunction(func) and not inspect.ismethod(func):
        # If it's not a standard function, we can't safely scan bytecode
        return False

    instructions = list(dis.get_instructions(func))
    
    for instr in instructions:
        # 1. Block unauthorized imports or global mutations
        if instr.opname in FORBIDDEN_OPCODES:
            raise PurityError(
                f"STTI-001 Violation: Forbidden opcode {instr.opname} detected in {func.__name__}. "
                "Tools must be deterministic and pure."
            )

        # 2. Block 'Dunder' attribute escapes (e.g., .__subclasses__())
        if instr.opname == "LOAD_ATTR":
            attr_name = instr.argval
            if attr_name in FORBIDDEN_ATTRIBUTES:
                raise PurityError(
                    f"STTI-001 Violation: Access to {attr_name} is forbidden. "
                    "Cannot verify provenance of escaped objects."
                )

    return True

def enforce_stti_invariant(response: Any, expected_type: str, source: str) -> None:
    """
    Final check: Ensures the returned object matches the ledger's notarization.
    Stops 'True vs 1' type-coercion bypasses.
    """
    actual_type = type(response).__name__
    if actual_type != expected_type:
        raise PurityError(
            f"STTI-001 Type Collision: Expected {expected_type} from {source}, "
            f"but found {actual_type}. Execution halted."
        )
