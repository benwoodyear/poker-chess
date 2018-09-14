import random
from collections import Counter
from operator import itemgetter
import itertools
import chess
import numpy as np


class Deal(object):
    """
    Class to deal out the cards in the shape of a chess board. Currently allows for either a random deal, where the
    cards are placed in any location, or a symmetrical deal where the cards mirror each other - eg. 4H would be opposite
    4D. In the future add predefined deals or other options.
    """
    def __init__(self, deal_type):
        self.deal_type = deal_type

    def deal_cards(self):
        deal_type = self.deal_type

        card_board = np.empty((8, 8), dtype=object)

        card_names = []
        suits = ['H', 'D', 'S', 'C']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        for i in suits:
            for j in values:
                card_names.append(j + i)

        if deal_type == 'random':
            random.shuffle(card_names)
            k = 0
            for i in range(8):
                for j in range(8):
                    if (i == 3 and j != 0 and j != 7) or (i == 4 and j != 0 and j != 7):
                        card_board[i, j] = '  '
                    else:
                        card_board[i, j] = card_names[k]
                        k += 1
            return card_board

        elif deal_type == 'symmetrical':
            hearts_and_spades = []
            clubs_and_diamonds = []
            # create a list of just the hearts and spades, then randomise
            for x in card_names:
                if x[-1] == 'H' or x[-1] == 'S':
                    hearts_and_spades.append(x)
            random.shuffle(hearts_and_spades)
            # mirror the hearts and spades list with clubs and diamonds
            for y in hearts_and_spades:
                if y[-1] == 'H':
                    clubs_and_diamonds.append(y[0] + 'C')
                else:
                    clubs_and_diamonds.append(y[0] + 'D')
            # lay the cards out on the board, hearts and spades at the top
            k = 0
            for i in range(4):
                for j in range(8):
                    if (i == 3 and j != 0 and j != 7) or (i == 4 and j != 0 and j != 7):
                        card_board[i, j] = '  '
                        card_board[7-i, j] = '  '
                    else:
                        card_board[i, j] = hearts_and_spades[k]
                        card_board[7-i, j] = clubs_and_diamonds[k]
                        k += 1
            return card_board


def choose_deal():
    """
    Prompts a user to enter either r or s, this can then be fed into the Deal class to chose the deal pattern.
    """
    deal_chosen = False
    while deal_chosen is False:
        deal_type = input('For a random or symmetrical card deal enter r/s: ')
        if deal_type == 'r':
            return 'random'
        elif deal_type == 's':
            return 'symmetrical'
        else:
            print('Please enter r or s.')


def make_piece_tracker():
    """
    Creates the hidden board which will be used to check poker hands. Makes an 8x8 matrix with lowercase pieces
    corresponding to black and uppercase for white.
    """
    black_row = np.array(['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'])
    black_pawns = np.repeat('p', 8)
    empty_row = np.repeat('/', 8)
    white_row = np.array(['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'])
    white_pawns = np.repeat('P', 8)
    hidden_piece_board = np.vstack((black_row, black_pawns, empty_row, empty_row,
                                    empty_row, empty_row, white_pawns, white_row))
    return hidden_piece_board


def chess_coords_to_matrix(chess_coord):
    """
    Converts a chess coordinate to a matrix coordinate.
    :param chess_coord: e4
    :return: (4, 5)
    """
    conversion = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
    i_coord = 8 - int(chess_coord[1])
    j_coord = int(conversion[chess_coord[0]]) - 1
    matrix_coord = (i_coord, j_coord)
    return matrix_coord


def piece_remover(piece_to_remove, piece_rank_white, piece_rank_black):
    """
    Searches through the list of pieces to remove a piece which has been taken. Also used to remove pawns after
    promotion.
    """
    if piece_to_remove == piece_to_remove.upper():
        for i in piece_rank_white:
            try:
                i.remove(piece_to_remove)
            except ValueError:
                pass
    else:
        for i in piece_rank_black:
            try:
                i.remove(piece_to_remove)
            except ValueError:
                pass


def update_piece_tracker(starting, ending, piece_board, piece_rank_white, piece_rank_black):
    """
    Takes the start and end matrix coordinates of a move, and checks whether this move leads to a piece being taken.
    If a piece is taken the piece_remover() function is called on the piece being taken. Then the chess piece matrix is
    updated with the piece at its new position and its original position empty.
    """
    if piece_board[ending] != '/':
        piece_remover(piece_board[ending], piece_rank_white, piece_rank_black)
        piece_board[ending] = piece_board[starting]
        piece_board[starting] = '/'
    else:
        piece_board[ending] = piece_board[starting]
        piece_board[starting] = '/'


def find_hand_selecting_pieces(piece_list):
    """
    This function first finds the best four pieces the player owns, then adds a pawn if there are any left and they
    haven't already been counted as part of the best four pieces.
    """
    def find_best_four_pieces(piece_list):
        best_pieces = []
        for i in piece_list:
            if len(best_pieces) == 4:
                return best_pieces
            elif len(i) + len(best_pieces) <= 4:
                for j in i:
                    best_pieces.append(j)
            elif len(i) + len(best_pieces) > 4:
                for j in range(4 - len(best_pieces)):
                    best_pieces.append(i[j])
        return best_pieces

    piece_selection = find_best_four_pieces(piece_list)
    # Check if there are any pawns left
    if len(piece_list[5]) == 0:
        pass
    # Making sure that pawns aren't being double counted
    elif piece_selection.count(piece_list[5][0]) == len(piece_list[5]):
        pass
    else:
        piece_selection.append(piece_list[5][0])
    return piece_selection


def hand_locations(board_matrix, pieces):
    """
    This function takes the numpy array where the locations of the pieces are logged and the locations of the pieces
    which will be considered to make the poker hand. It then searches the matrix for for pieces matching the names
    and returns the coordinates of that piece. These coordinates are stored as a tuple so they can be input
    into the card matrix to find the card at that location.
    """
    hand = find_hand_selecting_pieces(pieces)
    all_hand_locations = []
    for i in range(len(hand)):
        piece_locations = []
        array_positions = np.where(board_matrix == hand[i])
        positions_0 = array_positions[0]
        positions_1 = array_positions[1]
        positions = np.vstack((positions_0, positions_1))
        for j in range(np.size(positions, axis=1)):
            piece_locations.append(tuple(positions[:, j]))
        all_hand_locations.append(piece_locations)
    return all_hand_locations


def find_cards(piece_locations, card_deal):
    """
    Searches through the list containing the locations of the pieces making up the hand, then uses these coordinates
    to return the cards which are at those positions. These combinations will then be checked to find out the best
    poker hand possible.
    N.B. currently the piece locations are overwritten when this function is executed. This shouldn't be a problem, but
    would be good to change this at some point.
    """
    cards_at_locations = piece_locations
    for i in cards_at_locations:
        for j in range(len(i)):
            card = card_deal[i[j]]
            if card == '  ':
                i[j] = 0
            else:
                i[j] = card
    card_selection = []
    # Since pieces can be in the central spaces without cards, this needs to be taken into account. This instances are
    # entered as 0, which are then cleaned from the first list with a list comprehension.
    for i in range(len(cards_at_locations)):
        card_selection.append([])
        # Removing the cases of 0 from the lists
        card_selection[i] = [x for x in cards_at_locations[i] if x != 0]
    # Making sure there are no empty lists is the card list, as this causes itertools.product to return no combinations
    final_card_selection = [y for y in card_selection if y != []]
    print(final_card_selection)
    return final_card_selection


def unique_counter(x):
    """
    Takes a list x and counts how many times that value appears in the list.
    :param x: [2, 3, 4, 3, 3, 4]
    :return: [1, 3, 2, 3, 3, 2]
    """
    index_counts = []
    for i in x:
        index_counts.append(x.count(i))
    return index_counts


class PokerScorer(object):
    """
    This class will take a poker hand, then return a rank and also allow for comparisons between hands of the same rank.
    It initially splits the cards into suit and value components, before testing the hand using the various
    evaluating functions, which are executed within the hand_ranking() function.
    """

    def __init__(self, poker_hand):
        self.poker_hand = poker_hand

    def value_splitter(self):

        card_scores = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                       'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

        poker_hand = self.poker_hand
        hand_values = []
        for i in poker_hand:
            hand_values.append(card_scores[i[0]])
        hand_values.sort()
        return hand_values

    def suit_splitter(self):
        poker_hand = self.poker_hand
        hand_suits = []
        for i in poker_hand:
            hand_suits.append(i[-1])
        return hand_suits

    def final_rank_determiner(self):
        """
        Sorts the hand values first by descending value and then number of appearances.
        :param: [2, 5, 14, 5, 2]
        :return: [5, 5, 2, 2, 14]
        """
        hand_values = self.value_splitter()
        hand_values.sort(reverse=True)
        # Use the Counter function to sort the values by frequency of appearance
        hand_values.sort(key=Counter(hand_values).get, reverse=True)
        while len(hand_values) < 5:
            hand_values.append(0)
        return hand_values

    def suit_order_determiner(self):
        """
        This function is to return a sorted list of lists, with each sublist containing the value, suit and number of
        appearances of that value. These will be put into the matrix returned by the hand ranker, allowing it to be
        efficiently sorted. The winning hand can then be reconstructed form the numbers.

        This doesn't seem like a very efficient way at the moment - look at optimising.
        """
        poker_hand = self.poker_hand

        card_scores = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                       'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

        suit_numbers = {'C': 1, 'D': 2, 'H': 3, 'S': 4}

        separated_cards = []
        for i in poker_hand:
            separated_cards.append([card_scores[i[0]], suit_numbers[i[1]]])

        value_appearances = unique_counter([i[0] for i in separated_cards])

        values_suits_appearances = []
        for j in range(len(separated_cards)):
            values_suits_appearances.append(separated_cards[j] + [value_appearances[j]])

        ordered_data = sorted(values_suits_appearances, key=itemgetter(2, 0), reverse=True)
        ordered_suits = [k[1] for k in ordered_data]
        # To make sure all matrices are dimensionally consistent append zeroes to make up length to 5
        while len(ordered_suits) < 5:
            ordered_suits.append(0)
        return ordered_suits

    @staticmethod
    def straight_test(hand_values):
        if sum(np.ediff1d(hand_values)) == 4 and set(np.ediff1d(hand_values)) == {1}:
            return True
        elif sum(np.ediff1d(hand_values)) == 12 and set(np.ediff1d(hand_values)) == {1, 9}:
            return True
        else:
            return False

    @staticmethod
    def flush_test(hand_suits):
        if len(hand_suits) == 5 and len(set(hand_suits)) == 1:
            return True
        else:
            return False

    @staticmethod
    def quads_test(hand_values):
        if max(set(unique_counter(hand_values))) == 4:
            return True
        else:
            return False

    @staticmethod
    def full_house_test(hand_values):
        if set(unique_counter(hand_values)) == {2, 3}:
            return True
        else:
            return False

    @staticmethod
    def trips_test(hand_values):
        if max(set(unique_counter(hand_values))) == 3 and min(set(unique_counter(hand_values))) != 2:
            return True
        else:
            return False

    @staticmethod
    def two_pair_test(hand_values):
        if max(set(unique_counter(hand_values))) == 2 and \
                max(set(unique_counter(unique_counter(hand_values)))) == 4:
            return True
        else:
            return False

    @staticmethod
    def pair_test(hand_values):
        if max(set(unique_counter(hand_values))) == 2 and \
                max(set(unique_counter(unique_counter(hand_values)))) != 4:
            return True
        else:
            return False

    def hand_ranker(self):
        poker_hand = self.poker_hand
        poker_hand.sort(reverse=True)
        hand_values = self.value_splitter()
        hand_suits = self.suit_splitter()
        rank_placement = self.final_rank_determiner()
        suit_placement = self.suit_order_determiner()

        if self.straight_test(hand_values) is True and self.flush_test(hand_suits) is True:
            hand_score = [9] + rank_placement + suit_placement
            return hand_score
        elif self.quads_test(hand_values) is True:
            hand_score = [8] + rank_placement + suit_placement
            return hand_score
        elif self.full_house_test(hand_values) is True:
            hand_score = [7] + rank_placement + suit_placement
            return hand_score
        elif self.flush_test(hand_suits) is True and self.straight_test(hand_values) is False:
            hand_score = [6] + rank_placement + suit_placement
            return hand_score
        elif self.straight_test(hand_values) is True and self.flush_test(hand_suits) is False:
            hand_score = [5] + rank_placement + suit_placement
            return hand_score
        elif self.trips_test(hand_values) is True:
            hand_score = [4] + rank_placement + suit_placement
            return hand_score
        elif self.two_pair_test(hand_values) is True:
            hand_score = [3] + rank_placement + suit_placement
            return hand_score
        elif self.pair_test(hand_values) is True:
            hand_score = [2] + rank_placement + suit_placement
            return hand_score
        else:
            hand_score = [1] + rank_placement + suit_placement
            return hand_score


def possible_hand_combinator(all_cards):
    """
    Takes a list containing the lists of possible cards corresponding to the different pieces then works out the
    different possible combinations and eliminates duplicate hands and any illegitimate hands.
    """
    all_card_products = [p for p in itertools.product(*all_cards)]
    # This ensures that two of the same card haven't been included in the hand
    duplicate_cards_removed = [list(set(i)) for i in all_card_products]
    # This next step removes hands with just a different order of cards
    possible_hands = [list(i) for i in set(frozenset(i) for i in duplicate_cards_removed)]
    return possible_hands


def compare_best_hands(white_pieces, black_pieces, piece_matrix, card_matrix):
    """
    This function first uses the hand_locations() function to find where the relevant cards are located. Then this is
    fed into find_cards() which returns the cards at those locations. This in turn is passed to
    possible_hand_combinator() white works out which combinations of these cards form valid and unique hands.

    This list of possible hands is fed via a loop into the PokerScorer class, then hand_ranker() gives each
    hand a 'score' and puts the card values in order of importance, allowing hands to be compared and sorted.

    The matrix of hands is sorted, then the top value is taken, as this is the best hand from the selection. Finally
    the best hands for white and black are compared, and the winner is announced!
    """

    white_piece_location = hand_locations(piece_matrix, white_pieces)
    black_piece_location = hand_locations(piece_matrix, black_pieces)

    all_white_cards = find_cards(white_piece_location, card_matrix)
    all_black_cards = find_cards(black_piece_location, card_matrix)

    all_white_hands = possible_hand_combinator(all_white_cards)
    all_black_hands = possible_hand_combinator(all_black_cards)

    white_hand_list = []
    for x in all_white_hands:
        white_hand_list.append(PokerScorer(x).hand_ranker())
    white_hand_matrix = np.array(white_hand_list, dtype='<i4')
    # Use lexsort() to sort the the matrix from the 6th to the 1st column
    best_white_hand_matrix = white_hand_matrix[np.lexsort((white_hand_matrix[:, 5], white_hand_matrix[:, 4],
                                                           white_hand_matrix[:, 3], white_hand_matrix[:, 2],
                                                           white_hand_matrix[:, 1], white_hand_matrix[:, 0]))][:: -1]
    best_white_hand = best_white_hand_matrix[0, :]

    black_hand_list = []
    for x in all_black_hands:
        black_hand_list.append(PokerScorer(x).hand_ranker())
    black_hand_matrix = np.array(black_hand_list, dtype='<i4')
    # Use lexsort() to sort the the matrix from the 6th to the 1st column
    best_black_hand_matrix = black_hand_matrix[np.lexsort((black_hand_matrix[:, 5], black_hand_matrix[:, 4],
                                                           black_hand_matrix[:, 3], black_hand_matrix[:, 2],
                                                           black_hand_matrix[:, 1], black_hand_matrix[:, 0]))][:: -1]
    best_black_hand = best_black_hand_matrix[0, :]

    def matrix_to_hand(hand):
        """
        This takes the matrix row corresponding to a hand and transforms it into a string, with nice symbols
        representing the suits.
        """
        suit_to_symbol = {1: '\u2667', 2: '\u2662', 3: '\u2661', 4: '\u2664'}
        value_to_name = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                         10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        display_hand = ''
        for j in range(1, 6):
            if hand[j] == 0:
                pass
            else:
                if j != 5:
                    display_hand += value_to_name[hand[j]] + suit_to_symbol[hand[j+5]] + ' '
                else:
                    display_hand += value_to_name[hand[j]] + suit_to_symbol[hand[j+5]]
        return display_hand

    print('\n')
    print('White: ' + matrix_to_hand(best_white_hand), '    ', 'Black: ' + matrix_to_hand(best_black_hand))

    score_to_hand_name = {1: 'High Card', 2: 'One Pair', 3: 'Two Pairs', 4: 'Three of a Kind', 5: 'A Straight',
                          6: 'A Flush', 7: 'Full House', 8: 'Four of a Kind', 9: 'Straight Flush!'}

    for i in range(len(best_white_hand)):
        if best_white_hand[i] > best_black_hand[i]:
            print('White wins with: ' + score_to_hand_name[best_white_hand[0]])
            return 1
        elif best_white_hand[i] < best_black_hand[i]:
            print('Black wins with: ' + score_to_hand_name[best_black_hand[0]])
            return 2
        elif i == len(best_white_hand) - 1 and best_white_hand[i] == best_black_hand[i]:
            print('Draw - the hands are tied!')
        else:
            pass


def nice_card_layout(card_matrix):
    """
    Just swaps the suit letter for a unicode symbol and then prints the card matrix in a nice grid.
    """
    # To swap the letter representation of the suit to a pretty unicode symbol
    card_symbols = {'H': '\u2661', 'D': '\u2662', 'S': '\u2664', 'C': '\u2667'}

    board = []
    for i in range(8):
        row = ''
        for j in range(8):
            card = card_matrix[i][j]
            if card == '  ':
                nice_card = '  '
            else:
                nice_card = card[0] + card_symbols[card[1]]
            row += nice_card + ' '
        board.append(row)
    return board


def board_to_display(piece_matrix):
    """
    Creates a board with the unicode symbols for the pieces and dots for empty squares.
    """
    piece_to_number = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
    board = []

    for i in range(8):
        row = ''
        for j in range(8):
            piece = piece_matrix[i, j]
            if piece != '/':
                colour = piece.lower() == piece
                piece_number = piece_to_number[piece.upper()]
                row += chess.Piece(piece_number, colour).unicode_symbol() + ' '
            else:

                row += '\u2015 '
        board.append(row)
    return board


def move_generator(chess_board, piece_matrix, piece_rank_white, piece_rank_black):
    """
    This function prompts the players to enter the coordinates for the move, then executes this on the chess board. It
    updates the piece matrix so this can be searched to find which hands players have.
    """
    correct_move = False
    while correct_move is False:
        '''
        Make sure that the move entered is correct and doesn't lead to crashes.
        '''
        start_coord, end_coord = '  ', '  '

        # Check the input is in the right form - eg c2, c4
        while start_coord[0] not in 'abcdefgh' or start_coord[1] not in '12345678':
            start_coord = input('Enter where you want to move from: ')
            if len(start_coord) != 2:
                start_coord = '  '
            else:
                pass

        while end_coord[0] not in 'abcdefgh' or end_coord[1] not in '12345678':
            end_coord = input('Enter where you want to move to: ')
            if len(end_coord) < 2 or len(end_coord) > 3:
                end_coord = '  '
            elif len(end_coord) == 3 and end_coord[2] not in 'qrbn':
                end_coord = '  '
            else:
                pass

        if chess.Move.from_uci(start_coord + end_coord) in chess_board.legal_moves:

            chess_board.push(chess.Move.from_uci(start_coord + end_coord))

            # Updating the matrix and piece log
            start = chess_coords_to_matrix(start_coord[0:2])
            end = chess_coords_to_matrix(end_coord[0:2])

            # The dictionaries to add pieces from promotions
            white_piece_dict = {'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4}
            black_piece_dict = {'k': 0, 'q': 1, 'r': 2, 'b': 3, 'n': 4}

            # Look if a pawn if being promoted, then update the piece log to remove a pawn and add the new piece
            if len(end_coord) == 3:
                if piece_matrix[start] == 'P':
                    new_piece = end_coord[-1].upper()
                    piece_rank_white[white_piece_dict[new_piece]].append(new_piece)
                    piece_remover('P', piece_rank_white, piece_rank_black)
                else:
                    new_piece = end_coord[-1].lower()
                    piece_rank_black[black_piece_dict[new_piece]].append(new_piece)
                    piece_remover('p', piece_rank_white, piece_rank_black)
            else:
                pass

            # Checking if a piece is taken by the move, and updating the list of pieces if one is
            update_piece_tracker(start, end, piece_matrix, piece_rank_white, piece_rank_black)

            correct_move = True
        else:
            print('This is not a legal move.')


def game_function():
    """
    This function just pulls together all the previous classes and functions to run the game.
    """

    intro = input('If you would like a quick introduction to poker chess enter y: ')
    if intro == 'Y' or intro == 'y':
        print('''
              Welcome to poker chess! This game combines elements of both poker and chess to create an entirely new
              (and improved) experience. You can win by either a traditional checkmate victory or by being a set number
              of poker hands ahead.
              
              It works by dealing the cards out in the shape of a chess board with the central rectangle of twelve 
              squares missing. Hands are made up of the cards under your best four pieces and one pawn. So your initial 
              hand will be made from the cards under your king, queen, two rooks and one pawn. This changes during the 
              game as your pieces are taken.
              
              After black's move the best hand for each player is calculated. The player with the strongest hand adds a 
              point to their score. 
              
              You'll have to balance a chess advantage with holding onto a better hand, as a strong chess position will
              often correspond to a poor hand, due to the central squares being blank.
              
              You can choose between a random or a symmetrical deal, symmetrical is suggested for a more balanced game,
              as this way players start on even hands. Moves are entered using the chess coordinate system. The default
              hand score difference to win is 5, although this can be changed as you see fit.
              
              Any questions or bugs please contact benwoodyear@gmail.com.
                            
              Enjoy!
              ''')
        print('\n')
    else:
        pass

    white_hands_won = 0
    black_hands_won = 0

    # All the pieces of each colour, ranked in order to work out the poker hand
    white_pieces = [['K'], ['Q'], ['R', 'R'], ['B', 'B'], ['N', 'N'], ['P'] * 8]
    black_pieces = [['k'], ['q'], ['r', 'r'], ['b', 'b'], ['n', 'n'], ['p'] * 8]

    # Call the choose_deal() function to set the deal type, then enter this into the Deal class
    deal_type = choose_deal()
    new_deal = Deal(deal_type).deal_cards()

    # Create the board using the inbuilt python-chess option, then use make_piece_tracker() to create the matrix
    board = chess.Board()
    piece_matrix = make_piece_tracker()

    move_counter = 0

    victory_difference = 5

    # Allow the player to change the win limit between the hand scores
    change_hand_win_difference = input('Would you like the hand score win difference from 5? Enter y to alter. ')
    if change_hand_win_difference == 'y' or change_hand_win_difference == 'Y':
        victory_difference_check = False
        while victory_difference_check is False:
            victory_difference = input('''What poker score difference would you like to set as the victory limit?: ''')
            victory_difference_check = victory_difference.isdigit()
    else:
        pass

    # Keep looping while there is no checkmate and the hand score difference is less than the win cap
    while board.is_checkmate() is False and abs(white_hands_won - black_hands_won) < victory_difference:
        print('\n')

        x, y = board_to_display(piece_matrix), nice_card_layout(new_deal)
        for i in range(8):
            print(str(8 - i) + '| ' + x[i] + '  ' + y[i])
        print('  ----------------')
        print('   a b c d e f g h')
        print('\n')

        move_counter += 1

        if move_counter % 2 == 0:
            print('Black to move')
        else:
            print('White to move')
        move_generator(board, piece_matrix, white_pieces, black_pieces)

        # Check if the move counter is even, as this means black has just moved, then compare best hands if it is
        if move_counter % 2 == 0:
            hand_outcome = compare_best_hands(white_pieces, black_pieces, piece_matrix, new_deal)

            if hand_outcome == 1:
                white_hands_won += 1
            elif hand_outcome == 2:
                black_hands_won += 1
            else:
                pass

            print('White has ' + str(white_hands_won) + ' hands')
            print('Black has ' + str(black_hands_won) + ' hands')

    # Check which kind of victory has been achieved
    if board.is_checkmate() is False:
        if white_hands_won > black_hands_won:
            print('White wins a poker victory!')
        elif white_hands_won < black_hands_won:
            print('Black wins a poker victory!')
    else:
        print('Win by checkmate!')


game_function()
