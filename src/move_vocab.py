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

import chess

PROMOTION_PIECES = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN
)

# move_to_id, moves
def build_move_vocabulary() -> tuple[dict[str, int], list[str]]:
    # construct list of uci moves
    moves = []

    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            if from_square == to_square:
                continue
            
            move = chess.Move(
                from_square=from_square,
                to_square=to_square
            )
            moves.append(move.uci())
    
    promotion_ranks = (
        # white: rank 7 to rank 8
        (6, 7),

        # black: rank 2 to rank 1
        (1, 0)
    )

    for source_rank, destination_rank in promotion_ranks:
        for source_file in range(8):
            for file_offset in (-1, 0, 1):
                # left, forward, right

                destination_file = source_file + file_offset

                if not 0 <= destination_file < 8:
                    continue
                
                # construct from and to squares
                from_square = chess.square(
                    source_file, source_rank
                )

                to_square = chess.square(
                    destination_file, destination_rank
                )

                for promotion_piece in PROMOTION_PIECES:
                    move = chess.Move(
                        from_square=from_square,
                        to_square=to_square,
                        promotion=promotion_piece
                    )
                    moves.append(move.uci())
    
    move_to_id = {
        uci_move: move_id for move_id, uci_move in enumerate(moves)
    }

    return move_to_id, moves

if __name__ == "__main__":
    # test
    x, y = build_move_vocabulary()
    print(len(x), len(y))