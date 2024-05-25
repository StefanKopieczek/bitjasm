from dataclasses import dataclass
from enum import Enum
from lark import Lark, Transformer
from typing import Union
import sys


grammar = r"""    
    operation : alu_operation | load_operation | jump_operation

    jump_operation : unconditional_jump | comparison_jump | bitmask_jump

    alu_operation : BINARY_OPERATION argument "," argument "->" destination
        | UNARY_OPERATION argument "->" destination

    load_operation : load_command load_arg "->" load_dest

    load_command : "LOAD" | "LD" | "MOVE" | "MOV"

    load_dest : bit_preserving_load_dest | bit_replacing_load_dest
    bit_preserving_load_dest : "|" destination
    bit_replacing_load_dest : destination

    load_arg : shifted_load_arg | unshifted_load_arg
    shifted_load_arg : (unmasked_load_arg | masked_load_arg) LEFT_SHIFT number | (unmasked_load_arg | masked_load_arg) RIGHT_SHIFT number
    unshifted_load_arg : unmasked_load_arg | masked_load_arg
    masked_load_arg : argument "&" WORD_MASK    
    unmasked_load_arg: argument

    unconditional_jump : JUMP_COMMAND argument
    comparison_jump : JUMP_COMMAND argument "IF" comparison
    bitmask_jump : JUMP_COMMAND argument "IF" argument "HAS" ANY_ALL number BIT_VALUE

    comparison : argument comparator argument
    comparator: GT | LT | EQU | SGN_EQ

    destination : pointer_literal_dest | REGISTER | register_pointer
    argument : numeric_literal | pointer_literal_arg | REGISTER | register_pointer

    register_pointer : POINTER REGISTER
    pointer_literal_arg : POINTER MEMORY_ADDR
    pointer_literal_dest : numeric_literal
    numeric_literal : number

    number : BINARY_NUMBER | HEX_NUMBER | DECIMAL_NUMBER

    WORD_MASK : /(0b)?[0-1]{4}/
    LEFT_SHIFT : "<<"
    RIGHT_SHIFT : ">>"
    
    DECIMAL_NUMBER : /(0d)?[0-9]+/
    HEX_NUMBER : /0x[0-9a-f]+/
    BINARY_NUMBER : /0b[0-1]+/

    JUMP_COMMAND : "JUMP" | "JMP"
    ANY_ALL : "ANY" | "ALL"
    BIT_VALUE : "HIGH" | "LOW"

    GT : ">"
    LT : "<"
    EQU : "==" | "="
    SGN_EQ: "SGNEQ" | "SIGN EQ"

    UNARY_OPERATION : "NEG" | "NOT" | "FNEG"
    BINARY_OPERATION : "ADD" | "SUB" | "MUL" | "IMUL" | "DIV" | "IDIV" | "MOD" | "AND" | "OR" | "XOR" | "LSHIFT" | "RSHIFT" | "LASHIFT" | "LRSHIFT" | "LROT" | "RROT" | "FLADD" | "FLSUB" | "FMUL" | "FDIV"
    POINTER : "*"

    MEMORY_ADDR : /(0x)?[a-f0-9]{1,8}/
    REGISTER : "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I"| "J" | "FL" | "IE" | "IF" | "S" | "ALU_OUT"

    %ignore /[ \t\f\r()]+/
"""


class Register(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8
    J = 9
    FL = 10
    IE = 11
    IF = 12
    S = 13
    ALU_OUT = 14   


@dataclass
class Literal:
    value: int


@dataclass
class Pointer:
    ref: Union[Literal, Register] 


class AssemblyTransformer(Transformer):
    def __init__(self):
        self.registers = {
            "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
            "I": 8, "J": 9, "FL": 10, "IE": 11, "IF": 12, "S": 13, "ALU_OUT": 14
        }
        self.alu_operations = {
            "ADD": 0, "SUB": 1, "MUL": 2, "IMUL": 3, "DIV": 4, "IDIV": 5, "MOD": 6,
            "AND": 7, "OR": 8, "XOR": 9, "LSHIFT": 10, "RSHIFT": 11, "LASHIFT": 12,
            "LRSHIFT": 13, "LROT": 14, "RROT": 15, "FLADD": 16, "FLSUB": 17,
            "FMUL": 18, "FDIV": 19, "NEG": 20, "NOT": 21, "FNEG": 22
        }


def assemble(code):
    parser = Lark(grammar, start='operation')
    transformer = AssemblyTransformer()

    machine_code = []
    for line in code.strip().split('\n'):
        tree = parser.parse(line)
        machine_code_bin = transformer.transform(tree)
        print(machine_code_bin)
        machine_code_hex = f"{int(machine_code_bin, 2):08x}"
        machine_code.append(machine_code_hex)

    return '\n'.join(machine_code)


# Example usage
assembly_code = """
ADD A, B -> C
"""
#LD 0x1234 -> D
#JMP A IF B == C
#"""

machine_code = assemble(assembly_code)
print(machine_code)
