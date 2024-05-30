from dataclasses import dataclass, replace
from enum import IntEnum, auto, unique
from bitstring import Bits
from lark import Lark, Transformer, Token
from typing import Union, Optional
import sys


grammar = r"""
    program : assembly_line*
    assembly_line : (LABEL ":")? operation

    ?operation : alu_operation | load_operation | jump_operation

    ?jump_operation : unconditional_jump | comparison_jump | bitmask_jump

    alu_operation : binary_operation argument "," argument "->" destination
        | unary_operation argument "->" destination

    load_operation : LOAD_COMMAND load_arg "->" load_dest

    LOAD_COMMAND : "LOAD"i | "LD"i | "MOVE"i | "MOV"i

    ?load_dest : bit_preserving_load_dest | bit_replacing_load_dest
    bit_preserving_load_dest : "|" destination
    bit_replacing_load_dest : destination

    ?load_arg : shifted_load_arg | unshifted_load_arg
    shifted_load_arg : (unmasked_load_arg | masked_load_arg) LEFT_SHIFT number | (unmasked_load_arg | masked_load_arg) RIGHT_SHIFT number
    unshifted_load_arg : unmasked_load_arg | masked_load_arg
    masked_load_arg : argument "&" WORD_MASK
    unmasked_load_arg: argument

    unconditional_jump : JUMP_COMMAND jump_destination
    comparison_jump : JUMP_COMMAND jump_destination "IF"i comparison
    bitmask_jump : JUMP_COMMAND jump_destination "IF"i argument "HAS"i ANY_ALL argument BIT_VALUE

    jump_destination : argument | LABEL
    LABEL : /(?!(A|B|C|D|E|F|G|H|I|J|FL|IE|EF|S|ALU_OUT)$)[a-zA-Z_][a-zA-Z0-9_-]*/

    comparison : argument comparator argument sign_indication?
    comparator: GT | LT | EQU | GTE | LTE | NEQ
    sign_indication: SIGNED | UNSIGNED
    SIGNED : "SIGNED"i
    UNSIGNED : "UNSIGNED"i

    ?destination : pointer_literal | register | register_pointer
    ?argument : numeric_literal | pointer_literal | register | register_pointer

    register_pointer : POINTER register
    pointer_literal : POINTER memory_addr
    ?numeric_literal : number

    ?number : binary_number | hex_number | decimal_number

    WORD_MASK : /(0[Bb])?[0-1]{4}/
    LEFT_SHIFT : "<<"
    RIGHT_SHIFT : ">>"

    decimal_number: DECIMAL_NUMBER
    hex_number: HEX_NUMBER
    binary_number: BINARY_NUMBER
    DECIMAL_NUMBER : /(0d)?[0-9]+/
    HEX_NUMBER : /0[Xx][0-9a-fA-F]+/
    BINARY_NUMBER : /0[bB][0-1]+/

    JUMP_COMMAND : "JUMP"i | "JMP"i
    ANY_ALL : "ANY"i | "ALL"i
    BIT_VALUE : "HIGH"i | "LOW"i

    GT : ">"
    LT : "<"
    EQU : "==" | "="
    GTE : ">="
    LTE : "<="
    NEQ : "!="

    unary_operation : UNARY_OPERATION
    UNARY_OPERATION : "NEG"i | "NOT"i | "FLNEG"i
    binary_operation : BINARY_OPERATION
    BINARY_OPERATION : "ADD"i | "SUB"i | "MUL"i | "IMUL"i | "DIV"i | "IDIV"i | "MOD"i | "AND"i | "OR"i | "XOR"i | "LSHIFT"i | "RSHIFT"i | "LASHIFT"i | "LRSHIFT"i | "LROT"i | "RROT"i | "FLADD"i | "FLSUB"i | "FLMUL"i | "FLDIV"i
    POINTER : "*"

    memory_addr : /(0[xX])[a-fA-F0-9]{1,8}/
    register: REGISTER
    REGISTER : "A"i | "B"i | "C"i | "D"i | "E"i | "F"i | "G"i | "H"i | "I"i | "J"i | "FL"i | "IE"i | "IF"i | "S"i | "ALU_OUT"i

    %ignore /[ \t\f\r\n()]+/
"""


@unique
class Register(IntEnum):
    A = 0
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    FL = auto()
    IE = auto()
    IF = auto()
    S = auto()
    ALU_OUT = auto()

    def get_location_bits(self) -> Bits:
        return Bits(f'bin:1=0, uint:4={int(self)}')

    def word_length(self) -> int:
        return 0

    def __str__(self):
        return self._name_


@dataclass
class Literal:
    value: int

    def get_location_bits(self) -> Bits:
        return Bits('bin:1=0, hex:4=f')

    def get_value_bits(self) -> Bits:
        return Bits(f'uint:32={int(self.value)}')

    def word_length(self) -> int:
        return 1

    def __str__(self):
        return hex(self.value)


@dataclass
class Pointer:
    ref: Literal | Register

    def get_location_bits(self) -> Bits:
        return Bits('0b10000') | self.ref.get_location_bits()

    def word_length(self) -> int:
        return self.ref.word_length()

    def __str__(self):
        return '*' + str(self.ref)

Location = Literal | Register | Pointer


@dataclass
class LoadArg:
    ref: Location
    mask: int
    shift: int

    def __str__(self):
        s = str(self.ref)
        if self.mask != 0b1111:
            s = s + ' & ' + bin(self.mask)
        if self.shift:
            if self.mask != 0b1111:
                s = f'({s})'
            if self.shift > 0:
                s = f'{s} << {shift}'
            else:
                s = f'{s} >> {-shift}'
        return s


@dataclass
class LoadDest:
    ref: Location
    preserve_bits: bool

    def __str__(self):
        return ('|' if self.preserve_bits else '') + str(self.ref)


@dataclass
class LoadOperation:
    arg: LoadArg
    dest: LoadDest

    def get_opcode(self) -> Bits:
        return (
            Bits('bin=0001') +
            Bits('uint:10=0') +
            Bits(f'uint:4={self.arg.mask}') +
            Bits(f'int:3={self.arg.shift}') +
            Bits(f'bin={1 if self.dest.preserve_bits else 0}') +
            self.arg.ref.get_location_bits() +
            self.dest.ref.get_location_bits()
        )

    def get_words(self, label_dict: dict[str, Literal]) -> [Bits]:
        return [self.get_opcode()] + locations_to_words(label_dict, self.arg.ref, self.dest.ref)

    def word_length(self) -> int:
        return 1 + self.arg.ref.word_length() + self.dest.ref.word_length()

    def __str__(self):
        return f'LOAD {str(self.arg)} -> {str(self.dest)}'


@unique
class AluOperator(IntEnum):
    ADD = 0
    SUB = auto()
    MUL = auto()
    IMUL = auto()
    DIV = auto()
    IDIV = auto()
    MOD = auto()
    NEG = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    NOT = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    LASHIFT = auto()
    LRSHIFT = auto()
    LROT = auto()
    RROT = auto()
    FLADD = auto()
    FLSUB = auto()
    FLMUL = auto()
    FLDIV = auto()
    FLNEG = auto()

    def get_code_bits(self) -> Bits:
        return Bits(f'uint:5={int(self)}')

    def __str__(self):
        return self._name_


@dataclass
class AluOperation:
    operator: AluOperator
    arg1: Location
    arg2: Location
    dest: Location

    def get_opcode(self) -> Bits:
        return (
            Bits('uint:12=0')
            + self.arg1.get_location_bits()
            + self.arg2.get_location_bits()
            + self.dest.get_location_bits()
            + self.operator.get_code_bits()
        )

    def get_words(self, label_dict: dict[str, Literal]) -> [Bits]:
        return [self.get_opcode()] + locations_to_words(label_dict, self.arg1, self.arg2, self.dest)

    def word_length(self) -> int:
        return 1 + self.arg1.word_length() + self.arg2.word_length() + self.dest.word_length()

    def __str__(self):
        return f'{self.operator} {self.arg1}, {self.arg2} -> {self.dest}'


@dataclass
class LabelRef:
    label: str

    def __str__(self):
        return self.label

    def word_length(self) -> int:
        return 1

    def get_location_bits(self) -> Bits:
        # A label is just a literal.
        return Bits('bin=01111')


JumpDestination = Location | LabelRef


@dataclass
class UnconditionalJump:
    dest: JumpDestination

    def get_opcode(self) -> Bits:
        return (
            Bits('bin=0010000000000')
            + self.dest.get_location_bits()
            + Bits('bin=00000000000000')
        )

    def get_words(self, label_dict: dict[str, Literal]) -> [Bits]:
        return [self.get_opcode()] + locations_to_words(label_dict, self.dest)

    def word_length(self) -> int:
        return 1 + self.dest.word_length()

    def __str__(self):
        return f'JUMP {self.dest}'


@dataclass
class BitmaskJump:
    dest: JumpDestination
    arg: Location
    mask: Location
    is_or: bool
    seek_high: bool

    def get_opcode(self) -> Bits:
        return (
            Bits('bin=0010000000001')
            + self.dest.get_location_bits()
            + Bits('bin=00')
            + self.arg.get_location_bits()
            + self.mask.get_location_bits()
            + Bits(f'bin={1 if self.is_or else 0}')
            + Bits(f'bin={1 if self.seek_high else 0}')
        )

    def get_words(self, label_dict: dict[str, Literal]) -> [Bits]:
        return [self.get_opcode()] + locations_to_words(label_dict, self.dest, self.arg, self.mask)

    def word_length(self) -> int:
        return 1 + self.dest.word_length() + self.arg.word_length() + self.mask.word_length()

    def __str__(self):
        return (
            f'JUMP {self.dest} IF {self.arg} & {hex(self.mask)} HAS '
            + ('ANY' if self.is_or else 'ALL')
            + ('HIGH' if self.seek_high else 'LOW')
        )


@dataclass
class Comparison:
    arg1: Location
    arg2: Location
    operator: str
    is_signed: Optional[bool]

    def __str__(self):
        s = f'{self.arg1} {self.operator} {self.arg2}'
        if self.is_signed is not None:
            if self.is_signed:
                s += ' (signed)'
            else:
                s += ' (unsigned)'
        return s


@dataclass
class ComparisonJump:
    dest: JumpDestination
    comparison: Comparison

    def get_opcode(self) -> Bits:
        if self._can_use_sign_jump():
            return self._get_opcode_sign_jump()
        else:
            return self._get_opcode_true_comparison()

    def get_words(self, label_dict: dict[str, Literal]) -> [Bits]:
        if self._can_use_sign_jump():
            return [self.get_opcode()] + locations_to_words(label_dict, self.dest, self.comparison.arg1)
        else:
            return [self.get_opcode()] + locations_to_words(label_dict, self.dest, self.comparison.arg1, self.comparison.arg2)

    def _can_use_sign_jump(self) -> bool:
        match self.comparison.arg2:
            case Literal(0):
                return True
            case _:
                return False

    def word_length(self) -> int:
        result = 1 + self.dest.word_length() + self.comparison.arg1.word_length()
        if not self._can_use_sign_jump():
            result += self.comparison.arg2.word_length()
        return result

    def _get_opcode_sign_jump(self) -> Bits:
        assert self.comparison.is_signed is not False, f"Unsigned comparison to 0 not allowed in {self}"
        return (
            Bits('bin=0010000000011')
            + self.dest.get_location_bits()
            + Bits('bin=000000')
            + self.comparison.arg1.get_location_bits()
            + self._get_operation_bits()
        )

    def _get_opcode_true_comparison(self) -> Bits:
        assert self.comparison.is_signed is not None or self.comparison.operator in ['=', '==', '!='], f"Ambiguous comparison; please add (signed) or (unsigned) to the end of the line in {self}"
        return (
            Bits('bin=0010000000010')
            + self.dest.get_location_bits()
            + self.comparison.arg1.get_location_bits()
            + self.comparison.arg2.get_location_bits()
            + self._get_operation_bits()
            + Bits(f'bin={1 if self.comparison.is_signed else 0}')
        )

    def _get_operation_bits(self) -> Bits:
        match self.comparison.operator:
            case '<':
                return Bits('bin=100')
            case '>':
                return Bits('bin=001')
            case '==':
                return Bits('bin=010')
            case '=':
                return Bits('bin=010')
            case '!=':
                return Bits('bin=101')
            case '>=':
                return Bits('bin=011')
            case '<=':
                return Bits('bin=110')
            case _:
                raise Exception(f'Unknown comparison operator {self.comparison.operator} in {self}')

    def __str__(self) -> str:
        return f'JUMP {self.dest} IF {self.comparison}'


Operation = AluOperation | LoadOperation | UnconditionalJump | BitmaskJump | ComparisonJump


@dataclass
class AssemblyLine:
    operation: Operation
    label: Optional[str]

    def __str__(self):
        if self.label:
            return f'{self.label}: {self.operation}'
        else:
            return str(self.operation)


def locations_to_words(label_dict: dict[str, Literal], *locations: Union[Location, JumpDestination]) -> [Bits]:
    result: [Bits] = []
    for location in locations:
        match location:
            case LabelRef(label):
                assert label in label_dict, f'Unresolved label "{label}"'
                result.append(label_dict[label].get_value_bits())
            case Literal(_):
                result.append(location.get_value_bits())
            case Pointer(Literal(val)):
                result.append(Literal(val).get_value_bits())
            case _:
                pass
    return result


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
        super().__init__(visit_tokens=True)

    def __default__(self, data, children, meta):
        print(f'No attribute matching {data} [children={children}; meta={meta}]')
        return super().__default__(data, children, meta)

    def register(self, items):
        return Register[items[0]]

    def memory_addr(self, items):
        return Literal(int(items[0], 16))

    def hex_number(self, items):
        return Literal(int(items[0], 16))

    def binary_number(self, items):
        return Literal(int(items[0], 2))

    def decimal_number(self, items):
        return Literal(int(items[0], 10))

    def register_pointer(self, items):
        return Pointer(items[1])

    def pointer_literal(self, items):
        return Pointer(items[1])

    def binary_operation(self, items):
        return AluOperator[items[0]]

    def unary_operation(self, items):
        return AluOperator[items[0]]

    def alu_operation(self, items):
        return AluOperation(
            operator=items[0],
            arg1=items[1],
            arg2=items[2],
            dest=items[3]
        )

    def unmasked_load_arg(self, items):
        return LoadArg(
            ref = items[0],
            mask = 0xf,
            shift = 0
        )

    def masked_load_arg(self, items):
        return LoadArg(
            ref = items[0],
            mask = int(items[1], 2),
            shift = 0
        )

    def unshifted_load_arg(self, items):
        return items[0]

    def shifted_load_arg(self, items):
        shift = items[2].value
        if items[1] == '>>':
            shift = -shift

        return replace(items[0], shift=shift)

    def bit_replacing_load_dest(self, items):
        return LoadDest(items[0], preserve_bits=False)

    def bit_preserving_load_dest(self, items):
        return LoadDest(items[0], preserve_bits=True)

    def load_operation(self, items):
        return LoadOperation(items[1], items[2])

    def jump_destination(self, items):
        dest = items[0]
        if isinstance(dest, Register | Literal | Pointer):
            return dest
        elif isinstance(dest, Token):
            return LabelRef(str(dest))
        else:
            raise Exception(f'Unknown jump destination type {dest} (type {type(dest)}')

    def unconditional_jump(self, items):
        return UnconditionalJump(dest=items[1])

    def bitmask_jump(self, items):
        is_or = str(items[3]) == 'ANY'
        seek_high = str(items[5]) == 'HIGH'
        return BitmaskJump(
            dest = items[1],
            arg = items[2],
            is_or = is_or,
            mask = items[4],
            seek_high = seek_high,
        )

    def comparator(self, items):
        return items[0]

    def sign_indication(self, items):
        return items[0]

    def comparison(self, items):
        is_signed = None if len(items) == 3 else str(items[3]).lower() == 'signed'
        return Comparison(
            arg1=items[0],
            arg2=items[2],
            operator=str(items[1]),
            is_signed = is_signed
        )

    def comparison_jump(self, items):
        return ComparisonJump(
            dest=items[1],
            comparison=items[2]
        )


    def assembly_line(self, items):
        if len(items) > 1:
            return AssemblyLine(operation=items[1], label=str(items[0]))
        else:
            return AssemblyLine(operation=items[0], label=None)


    def program(self, items):
        return items


def build_label_dict(assembly_lines: [AssemblyLine], offset=0) -> dict[str, Literal]:
    label_dict: dict[str, Literal] = {}
    words_seen = 0
    for line in assembly_lines:
        if line.label is not None:
            label_dict[line.label] = Literal(words_seen + offset)
        words_seen += line.operation.word_length()

    return label_dict


def assemble(code, mode='hex'):
    parser = Lark(grammar, start='program')
    transformer = AssemblyTransformer()

    machine_code = []
    tree = parser.parse(code.strip())
    parsed_lines = transformer.transform(tree)
    label_dict = build_label_dict(parsed_lines)

    line_number = 0
    for assembly_line in parsed_lines:
        operation = assembly_line.operation
        for idx, word in enumerate(operation.get_words(label_dict)):
            original_code = assembly_line if idx == 0 else None
            output = format_line(word, line_number, original=original_code, mode=mode)
            print(output)
            line_number += 1


def format_line(word: Bits, line_number: int, original : Optional[AssemblyLine] = None, mode: str ='hex'):
    if mode == 'hex':
        comment = f'  # {original}' if original else ''
        return f'0x{word.hex}{comment}'
    elif mode == 'vivado':
        comment = f'  // {original}' if original else ''
        return f'        flipflops[{line_number}] = 32\'h{word.hex};{comment}'
    else:
        raise Exception(f'Unknown format: "{mode}"')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--vivado':
        mode = 'vivado'
    else:
        mode = 'hex'

    assemble(sys.stdin.read(), mode=mode)
