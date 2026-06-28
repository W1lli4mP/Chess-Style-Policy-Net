"""
ID each and every UCI move

plan
fixed exhaustive vocabulary instead of one with only legal moves
why?: can mask illegal moves during inference
vocab must be fixed for policy logits to have a consistent dimension

uci ::= <square1> <square2> <promotion>?

for every square mapped to another square other than itself:
64 x 63 = 4032

4 promotion types

ways to promote:
8 pawns are able to go straight = 8
6 middle pawns are able to go left and right = 12
2 edge pawns are able to go left or right = 2
22 ways x 4 promotion types x 2 colours/directions = 176

thus, the vocab would contain 4032 + 176 = 4208 moves
"""

def build_move_vocabulary():
    pass

def move_to_id(uci: str) -> int:
    pass

def id_to_move(id: int) -> str:
    pass

