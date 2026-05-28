# Policy Network Project

this markdown contains my rough notes for now

## Main question
- how should a supervised policy network model a chess player's style when the player's behaviour changes significantly over time?

## Phases
Phase 1
- shared trunk + shared policy head + shared move-time head + time-control conditioning

Phase 2
- add bullet-specific head if bullet behaviour diverges strongly

Phase 3
- optionally add separate bullet/blitz/rapid heads

## Optional features
- including an auxiliary head dedicated for openings

## Requirements
## Python version
- Python 3.11
### Libraries
| Library      | Justification                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------- |
| numpy        | to handle data for the model                                                                      |
| torch        | to create and train the main components of the model                                              |
| python-chess | to handle PGNs from extracted chess.com data                                                      |
| requests     | to extract data from the chess.com API                                                            |
| pyarrow      | to read and write parquet files efficiently                                                       |
| pandas       | to filter data and work with parquet files                                                        |
| matplotlib   | OPTIONAL: could be useful for plotting training graphs                                            |
| tqdm         | OPTIONAL: could be useful for providing progress bars for downloading, preprocessing and training |

## Goal
- the goal is to train a supervised behavioural policy network that imitates my Chess.com playstyle, including both move choice and move speed
- the model is not intended to play the objectively best move; only intended to predict what I will probably play in a given board position and time situation

## Non-goals
- first version will not
	- use MCTS
	- use engine labels
	- optimise for maximum playing strength
- should preserve human-like variation by sampling from the move distribution rather than always choosing the argmax

## Limitations
- model imitates my historical decisions, including mistakes
- older games may not represent my current playstyle
- Chess.com clock comments could be noisy due to premoves, lag and increment
- rare positions and rare legal moves may be poorly learned
- model is not designed to find the objectively strongest move
- if trained on all eras equally, the model may average different versions of my playstyle

## Dataset strategy
- my playstyle has changed significantly over the dataset so older games may not represent my current style

possible strategies
- use all games for optional pretraining
- fine-tune mainly on recent/current-strength games
- use recency weighting so recent games affect the model more
- include my rating at the time of the game as a context feature
- validate/test mainly on recent games if the goal is current playstyle replication

## AlphaZero pipeline
AlphaZero pipeline (as an example for referencing/inspiration)
- current board
- tensor encoder
	- convert current position into 8x8 feature planes
- policy/value neural network
	- shared residual CNN trunk
	- policy head: predicts promising moves
	- value head: evaluates the position
- MCTS guided by policy priors and value estimates
	- uses policy head output as priors for move exploration
	- uses value head output to evaluate leaf nodes
	- returns improved move probabilities via visit counts
- move selector
	- chooses the final move from MCTS visit counts

MCTS will not be used in the first version because it would push the system toward stronger move search, while the goal is to replicate my playstyle
## Architecture
### Board branch
board tensor -> ResNet CNN trunk -> board embedding
- board tensor
	- encoding of the board's pieces and information
### Context branch
context vector -> small MLP -> context embedding
- context vector
	- selected non-board information from parsed parquet row converted into numbers the NN can use
	- contains
		- time control bucket: bullet/blitz/rapid
		- base time seconds
		- increment seconds
		- my clock remaining
		- opponent clock remaining
		- my clock percentage remaining
		- opponent clock percentage remaining
		- move number/ply
		- my colour
		- rating difference
		- recent move time history
		- opponent rating
		- current rating at time of game
- small MLP
	- transforms raw metadata into a useful learned representation
	- why use it?: raw context vector is small, differently scaled and not yet transformed
### Shared embedding
board embedding + context embedding -> shared embedding
- embedding
	- a learned numeric representation of something
	- a vector of numbers that the NN produces internally
- board embedding
	- "what is happening on the board?"
- context embedding
	- "what situation are you in?"
- shared embedding
	- "given this board and this situation, what would you play?"
### Output heads
shared embedding -> policy head -> policy logits
shared embedding -> move-time head -> move-time logits
- policy head
	- small MLP that answers "what move would I play?"
- move-time head
	- small MLP that answers "how quickly would I play?"
	- outputs move-time represented as buckets
		- 0: premove/instant, 0-0.3s
		- 1: reflex, 0.3-1s
		- 2: fast, 1-3s
		- 3: think, 3-8s
		- 4: long think, 8s+

speed modelling
- phase of game/approximate phase

## Board tensor encoding
- 12 piece planes
	- 6 piece types x 2 colours
- 6 game-state planes
	- 1 side-to-move plane
	- 4 castling-rights planes
	- 1 en passant target plane
- 12 + 6 = 18
- so the board tensor is `18 x 8 x 8`
## Model Specifications
### Context MLP
- purpose
	- to transform raw metadata from the chess parquets into a useful learned representation
- in: context vector (`context_dim`)
	- `context_dim` is the number of numeric context features after encoding
- out: context embedding (64)
- `context_dim` -> 64 -> 64 -> 64

|            | Hidden 1 | Hidden 2 | Output |
| ---------- | -------- | -------- | ------ |
| Size       | 64       | 64       | 64     |
| Activation | ReLU     | ReLU     | none   |
- no direct loss
	- trained indirectly through policy loss and move-time loss
### Residual CNN
- in: board tensor (18 x 8 x 8)
- out: board embedding (256)
- 18 x 8 x 8 -> 4 res blocks -> 256
- res block
	- two 3x3 convolutions
	- 64 channels
	- padding 1
	- one skip connection
- diagram handwritten
- no direct loss
	- trained indirectly through policy loss and move-time loss

### Policy MLP
- policy head
- in: shared embedding (320)
	- board embedding + context embedding
	- 256 + 64 = 320
- out: policy logits (`num_moves`)
	- raw scores for each possible move
- 320 -> 256 -> `num_moves`
	- `num_moves` from UCI-vocabulary encoding

|            | Hidden | Output      |
| ---------- | ------ | ----------- |
| Size       | 256    | `num_moves` |
| Activation | ReLU   | none        |
- loss: cross-entropy over policy logits
	- trained directly through policy loss

### Move-time MLP
- move-time head
- in: shared embedding (320)
- out: move-time logits (5)
- 320 -> 64 -> 5
	- 5 from time buckets:
		- 0: premove/instant, 0-0.3s
		- 1: reflex, 0.3-1s
		- 2: fast, 1-3s
		- 3: think, 3-8s
		- 4: long think, 8s+

|            | Hidden | Output |
| ---------- | ------ | ------ |
| Size       | 64     | 5      |
| Activation | ReLU   | none   |
- loss: cross-entropy over move-time logits
	- trained directly through move-time loss

## Data
- extracted from: https://api.chess.com/pub/player/nerf_ee/games/archives
- parquets
	- is a file format used to store large datasets
	- instead of storing data row by row, it stores data column by column
- potential organisation
	- could use SQLite since it enables
		- filtering
		- searching
		- debugging
		- avoiding duplicate games

## How is chess.com data stored
- game archives
	- a JSON object with one field `archives`
	- `archives`
		- an array of HTTPS links to monthly games archive endpoint
- monthly games archive endpoint
	- a JSON object with one field `games`, containing a month's worth of games
	- `games: Game[]`
		- a JSON array encapsulating data for one game
		- `url: str`
			- chess.com URL of the game
		- `pgn: str`
			- portable game notation of the game
		- `time_control: str`
			- selected time control in seconds
		- `end_time: int`
			- standard Unix timestamp representing specifically when the game concluded
		- `rated: bool`
			- indicates if the game was rated
		- `tcn: str`
			- turn compressed notation of the game
		- `uuid: str`
			- unique ID of the game
		- `initial_setup: str`
			- FEN string of the initial position
		- `fen: str`
			- FEN string of the final position
		- `time_class: str`
			- time control as `"bullet"`, `"blitz"`, `"rapid"` or `"daily"`
		- `rules: str`
			- indicates what variant was used as `"chess"`, `"chess960"`, `"bughouse`, `"kingofthehill"`, `"threecheck"` or `"crazyhouse"`
		- `white: Player`
			- `rating: int`
				- ELO rating of the player
			- `result: str`
				- win condition as `"win"`, `"resigned"`, `"stalemate"`, `"checkmated"`, `"repetition"`, `"abandoned"`, `"timeout"`, `"insufficient"`, `"timevsinsufficient"`, `"50move"`, `"agreed"` or `"threecheck"`
			- `@id: str`
				- chess.com link to player's profile
			- `username: str`
				- player's username
			- `uuid: str`
				- unique ID
		- `black: Player`
			- `rating: int`
				- ELO rating of the player
			- `result: str`
				- win condition as `"win"`, `"resigned"`, `"stalemate"`, `"checkmated"`, `"repetition"`, `"abandoned"`, `"timeout"`, `"insufficient"`, `"timevsinsufficient"`, `"50move"`, `"agreed"` or `"threecheck"`
			- `@id: str`
				- chess.com link to player's profile
			- `username: str`
				- player's username
			- `uuid: str`
				- unique ID
		- `eco (OPTIONAL): str`
			- chess.com link to opening played

actual `Games[]` data that will be included:

| field         | type   | justification                                                                                      |
| ------------- | ------ | -------------------------------------------------------------------------------------------------- |
| pgn           | str    | to view the moves played                                                                           |
| time_control  | str    | gives the specific time control in seconds                                                         |
| end_time      | int    | to view the temporal information of when the game ended                                            |
| rated         | bool   | maybe; could add stronger weighting for rated games                                                |
| uuid          | str    | allows games to be uniquely identified, enabling the dataset to be extended with future games/data |
| initial_setup | str    | maybe; if bizarre custom positions were used                                                       |
| fen           | str    | maybe; just shows the final position                                                               |
| time_class    | str    | to find out the time control                                                                       |
| rules         | str    | only `"chess"` variants should be used                                                             |
| white         | Player | necessary information                                                                              |
| black         | Player | necessary information                                                                              |

actual `Player` data that will be used:

| field    | type | justification                                              |
| -------- | ---- | ---------------------------------------------------------- |
| rating   | int  | to allow ratings to have different weightings on the model |
| result   | str  | to differentiate the win conditions and to who             |
| username | str  | to differentiate between players                           |

## Move-time extraction
- move time is estimated through PGN comments
- so it is indirectly given through the Chess.com public dataset
- for each move
	- read the player's clock before and after the move
	- account for increment if available
	- estimate seconds spent on move
	- convert seconds spent into a move-time bucket

## Data filtering
include games where
- `rules == "chess"`
- `time_class` is `"bullet"`, `"blitz"` or `"rapid"`
- I am either white or black
- PGN is available and parseable
- the game has enough moves to produce useful training examples

exclude
- daily games
- variants such as chess960, bughouse, king of the hill, three-check and crazyhouse
- abandoned games with very few moves
- games with missing or unparseable PGN
- games without usable clock comments if training the move-time head

## How is loss calculated
- policy logits -> policy loss
- move-time logits -> move-time loss
- total loss
	- `policy_loss + λ * move_time_loss`
	- `λ = 0.2-0.5`
	- λ is a weighting factor to control how much move-time loss matters compared with policy/move loss

## Training setup
recency weighting
- using a sample weight
	- last 6 months: 1.0
	- 6-18 months: 0.7
	- older: 0.4
- this will be useful for openings (changed over time)

training targets
- input
	- board tensor
	- context vector
- output
	- move played by me
	- move-time bucket

training examples
- use every position where it is my turn to move
- target is the move I actually played
- opponent moves are only used to reconstruct the game state

move encoding
- use fixed move vocabulary
- map UCI moves to class IDs
- policy head outputs one logit per move class
- illegal moves are masked at inference

data splitting
- normal splitting
	- train: 70-80% of games
	- validation: 10-15% of games
	- test: 10-15% of games
- chronological split
	- train: older games
	- validation: more recent games
	- test: newest games

## Move vocabulary
- all moves are represented in UCI format
- each move in the vocabulary is assigned a fixed class ID
- promotion moves are included
- the policy head outputs one logit per move class
- illegal moves are masked before Softmax/sampling
- the same vocabulary is reused for training and inference

## Training settings
- batch size: 64
- optimiser: AdamW
- learning rate: 0.001 or 0.0003
- weight decay: 0.0001
- model trained end-to-end
## Inference
### Legal move masking
- model outputs logits for all move classes
- illegal moves are masked

### Sampling temperature
- during inference, not training
- temperature controls how deterministic the move choice is
- low temperature
	- more consistent, chooses top moves more often
- high temperature
	- more varied, more human/random

### Final move choice
- legal move probabilities are computed with Softmax
- move is sampled/chosen from legal moves
- policy logits -> mask illegal moves -> apply temperature -> Softmax -> sample move

## Inference Pipeline
- current board + time/context information
- board tensor + context vector
- policy network
- policy logits for every move in the move vocabulary
- mask illegal moves
- divide logits by temperature
- Softmax over legal moves
- sample one move from the probability distribution

## Evaluation
### Policy evaluation
- policy top-1 accuracy
- policy top-3 accuracy
- policy top-5 accuracy
- negative log likelihood of my actual moves

### Move-time evaluation
- move-time bucket accuracy
- average move-time bucket error
- confusion matrix between predicted and actual time buckets

### Breakdown evaluation
- evaluate separately for bullet, blitz, rapid
- evaluate separately for early, middle, current
- evaluate by rating bucket
- evaluate by game phase: opening, middlegame and endgame

### Qualitative evaluation
- sample moves from test positions
- compare sampled moves to my real choices
- check whether the model feels like my playstyle rather than just a legal move generator

## Era definition
- possible era splits
	- date-based
		- first 1-2 years, middle period, recent period
	- rating-based
		- beginner/intermediate/current rating ranges
	- hybrid
		- use both date and rating to define eras
- final era boundaries should be chosen after dataset analysis

## Dataset analysis
before deciding final splits/eras, inspect:
- number of games per month
- number of games per time class per month
- rating over time
- number of usable positions per month
- number of usable positions per era
- average move time per time class
- average move time by era
- distribution of move-time buckets
- distribution of bullet/blitz/rapid across eras
why?
- era boundaries should be based on actual shape of the dataset rather than arbitrary guesses
## Modelling strategies
| Experiment                                  | Description                                                             | What it will teach me                                                            |
| ------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| A. all-history model                        | train one model on every usable game, with rating/time/context features | - baseline modelling<br>- context conditioning<br>- large mixed dataset handling |
| B. recent-only model                        | train only on recent/current-strength games                             | - dataset curation<br>- distribution shift<br>- overfitting vs relevance         |
| C. era-specific models                      | train separate early/middle/current models                              | - temporal drift<br>- model comparison<br>- data segmentation                    |
| D. shared trunk + era heads                 | one shared trunk, separate policy/time heads per era                    | - multi-task learning<br>- parameter sharing<br>- modular architecture           |
| E. all-history pretrain -> recent fine-tune | train on all games, then fine-tune on current games                     | - transfer learning<br>- fine-tuning                                             |
| F. one model with era/rating conditioning   | add rating/era as input context                                         | - conditional modelling<br>- representation learning                             |
experimental order
1. dataset analysis
2. simple baseline model
3. all-history model
4. recent-only model
5. pretrain + fine-tune model
6. era-conditioned model
7. shared trunk + era-specific heads
8. compare all results

## Milestone 1
- download and parse Chess.com games
- create a parquet dataset of training positions
- train one shared policy + move-time model
- use legal move masking and temperature sampling at inference
- evaluate top-k move accuracy and move-time bucket accuracy