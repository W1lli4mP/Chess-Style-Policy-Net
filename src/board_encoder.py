# encode the board features from training row into a tensor
# so it is tangible for the CNN

import pandas as pd
import chess
import torch

def encode_board():
    training_rows = pd.read_parquet("../data/processed/training_positions.parquet")

    for training_row in training_rows:
        fen_str = training_row["fen_before_move"]

        son = chess.Board(fen_str) #! son im crine

        #* PIECE INFO

        #! ...

        #* GAME INFO

        # side to move: chess.WHITE | chess.BLACK
        side_to_move = son.turn

        # castling rights
        white_can_castle_kingside = son.has_kingside_castling_rights(chess.WHITE)
        white_can_castle_queenside = son.has_queenside_castling_rights(chess.WHITE)
        black_can_castle_kingside = son.has_kingside_castling_rights(chess.BLACK)
        black_can_castle_queenside = son.has_queenside_castling_rights(chess.BLACK)

        # en passant target: Square | None
        en_passant_target = son.ep_square

        encode_game_info()

    # ...


    # return tensor x where
    # BSx18x8x8
    # x = (batch_size, channels, height, width)

def encode_game_info(
    side_to_move,
    white_can_castle_kingside,
    white_can_castle_queenside,
    black_can_castle_kingside,
    black_can_castle_queenside,
    en_passant_target
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    en_side_to_move = torch.tensor(
        data=None,
        dtype=float
    )

    en_side_to_move = torch.ones(8, 8) if side_to_move == chess.WHITE else torch.zeros(8, 8)

    