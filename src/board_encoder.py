# encode the board features from training row into a tensor
# so it is tangible for the CNN

import pandas as pd
import chess
import torch

def encode_board():
    training_rows = pd.read_parquet("../data/processed/training_positions.parquet")

    for _, training_row in training_rows.iterrows():
        fen_str = training_row["fen_before_move"]

        son = chess.Board(fen_str)

        #* PIECE INFO

        #! ...
        piece_info_planes = None

        #* GAME INFO
        game_info_planes = encode_game_info(son)

    # ...


    # return tensor x where
    # BSx18x8x8
    # x = (batch_size, channels, height, width)

# 6 planes representing the game info (6x8x8)
def encode_game_info(board: chess.Board) -> torch.Tensor:
    

    enc_side_to_move = torch.full(
        (8, 8),
        fill_value=float(board.turn == chess.WHITE),
        dtype=torch.float32
    )
    
    enc_white_can_castle_kingside = torch.full(
        (8, 8),
        fill_value=float(board.has_kingside_castling_rights(chess.WHITE)),
        dtype=torch.float32
    )

    enc_white_can_castle_queenside = torch.full(
        (8, 8),
        fill_value=float(board.has_queenside_castling_rights(chess.WHITE)),
        dtype=torch.float32
    )

    enc_black_can_castle_kingside = torch.full(
        (8, 8),
        fill_value=float(board.has_kingside_castling_rights(chess.BLACK)),
        dtype=torch.float32
    )

    enc_black_can_castle_queenside = torch.full(
        (8, 8),
        fill_value=float(board.has_queenside_castling_rights(chess.BLACK)),
        dtype=torch.float32
    )

    enc_en_passant_target = torch.zeros((8, 8), dtype=torch.float32)
    if board.has_legal_en_passant() and board.ep_square is not None:
        enc_en_passant_target[
            chess.square_rank(board.ep_square),
            chess.square_file(board.ep_square)
        ] = 1.0

    return torch.stack(
        [
            enc_side_to_move,
            enc_white_can_castle_kingside,
            enc_white_can_castle_queenside,
            enc_black_can_castle_kingside,
            enc_black_can_castle_queenside,
            enc_en_passant_target
        ]
    )