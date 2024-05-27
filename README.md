# BitJASM

An assembler for the BitJam architecture. 

Documentation and a reference implementation for BitJam are coming soon.

## Usage:

`python bitjasm.py < my_program.jasm`

The assembled program is output to STDOUT in ASCII format with comments. A future BitJASM release will allow for raw binary output.

### Example:

```
➜ cat ~/pi_squared.jasm
LOAD 0x3f800000 -> B
LOAD B -> C
LOAD B -> I

LOAD B -> D
FLADD D, D -> D
FLMUL D, D -> E
FLADD D, E -> E

loop:
    FLADD C, I -> C
    LOAD C -> D
    FLMUL D, D -> D
    FLDIV I, D -> D
    FLADD B, D -> B
    FLMUL B, E -> A
JUMP loop

➜ python bitjasm.py < ~/pi_squared.jasm
0x1003c3c1  # LOAD 0x3f800000 -> B
0x3f800000
0x1003c042  # LOAD B -> C
0x1003c048  # LOAD B -> I
0x1003c043  # LOAD B -> D
0x00018c72  # FLADD D, D -> D
0x00018c94  # FLMUL D, D -> E
0x00019092  # FLADD D, E -> E
0x00012052  # loop: FLADD C, I -> C
0x1003c083  # LOAD C -> D
0x00018c74  # FLMUL D, D -> D
0x00040c75  # FLDIV I, D -> D
0x00008c32  # FLADD B, D -> B
0x00009014  # FLMUL B, E -> A
0x2003c000  # JUMP loop
0x00000008
```
