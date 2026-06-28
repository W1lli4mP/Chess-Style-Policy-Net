# encode the board features from training row into a tensor
# so it is tangible for the CNN

import pandas as pd
import chess
import torch

#! implement custom pytorch dataset class to use leverage batch encoding
def encode_boards():
    training_rows = pd.read_parquet("../data/processed/training_positions.parquet")

    for _, training_row in training_rows.iterrows():
        fen_str = training_row["fen_before_move"]

        son = chess.Board(fen_str)
        encode_board(son)

def encode_board(board: chess.Board):
    piece_info_planes = encode_piece_info(board) # (12, 8, 8)
    
    game_info_planes = encode_game_info(board) # (6, 8, 8)

    encoded_board = torch.cat(
        [piece_info_planes, game_info_planes],
        dim=0
    )

    # validate encoded board dimensions
    assert encode_board.shape == (18, 8, 8)

    return encoded_board

    # BSx18x8x8
    # x = (batch_size, channels, height, width)

# 12 planes representing the piece info (12x8x8)
def encode_piece_info(board: chess.Board) -> torch.Tensor:
    """
    plan
    initialise all piece planes
    iterate through board until piece_count() == 0
    update while iterating
    """

    planes = torch.zeros((6, 8, 8), dtype=torch.float32)

    # composite key of piece type and piece colour
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,

        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square, piece in board.piece_map().items():
        channel = piece_to_channel[(piece.piece_type, piece.color)]
        row = chess.square_rank(square)
        col = chess.square_file(square)
        planes[channel][row][col] = 1.0
    
    return planes

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